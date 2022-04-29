import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.signal as sps
import astropy.timeseries as ts
from astropy.io import fits
import os, sys, time, argparse, glob
from spinneret import *

parser = argparse.ArgumentParser(description='Run spinneret on every light curve in a given TESS sector')
parser.add_argument('-s', '--sector', required=True, type=int, help='TESS sector')

params = parser.parse_args()

sector = params.sector

id_prepend = 'tic'
file_append = 'tess'
cadence = 1/24/30 # 2min

# directorymaker('figs')

if len(str(sector)) == 1:
    opensec = '0' + str(sector)
else:
    opensec = str(sector)

dir_name = f's{opensec}results'
directorymaker(dir_name)

os.chdir(f'/data/shared_data/TESS/LightCurves/sector{opensec}')
target_list = glob.glob('*.fits')

# failed = []

for filename in target_list:

    try:
        hdu = fits.open(filename)

        table = hdu[1].data
        time = table['TIME']
        flux = table['PDCSAP_FLUX']

        extra = hdu[0].header
        teff = extra['TEFF']
        logg = extra['LOGG']
        tid = extra['TICID']

        hdu.close()

        minfreq = 1/(time[-1] - time[0])

        time, flux = nancleaner2d(time, flux)
        time, flux = clip(time, flux, 3) #3 sigma clip
        flux = lk.LightCurve(time=time, flux=flux).normalize().flux.value - 1

        target = Spinner(time, flux, teff, logg)

        freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
        target.ls_one_term(freq, ps)

        freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
        target.ls_two_term(freq, ps)

        lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, cadence, width=16)
        target.acf(lags, acf)

        # fig1 = target.diagnostic_plot(heading=f'TIC {tid}')
        # figsaver(fig1, f'TIC{tid}_{file_append}.png')

        filemaker(target, tid, 0, filename=f'{id_prepend}{tid}_{file_append}.csv', filepath=f'/home/icolman/data/spinneret/s{opensec}results')


        print(f'{tid} done')

    except:
        failed.append(tid)

os.chdir('/home/icolman/data/spinneret/s{opensec}results')
np.savetxt(failed, f's{opensec}failed.dat')
