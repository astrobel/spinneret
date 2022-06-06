import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.signal as sps
import astropy.timeseries as ts
from astropy.io import fits
import os, sys, time, argparse
from spinneret import *

parser = argparse.ArgumentParser(description='Process a Kepler quarter and TESSified sector using spinneret')
parser.add_argument('-f', '--filename', required=True, type=str, help='FITS file name')
# parser.add_argument('-i', '--targetid', required=True, type=int, help='KIC or TIC ID')
parser.add_argument('-s', '--rfset', required=False, default='test', type=str, choices=['test', 'train', 'validate'], help='test, train, or validate?')
parser.add_argument('-r', '--rotating', required=False, default='yes', type=str, choices=['yes', 'no', 'unknown'], help='Is the star rotating?')
parser.add_argument('-m', '--mission', required=False, default='TESS', type=str, choices=['TESS', 'TESSlike', 'Kepler'], help='TESS(like) sector or Kepler quarter?')
# parser.add_argument('-n', '--number', required=False, default=14, type=int, help='TESS sector or Kepler quarter; enter 14 for TESSlike Kepler data')
# needs an on/off switch for figures, defaulting to off

params = parser.parse_args()

filename = params.filename
# tid = params.targetid
tid = int(filename.split('-')[2])
forestset = params.rfset
rotating = params.rotating
mission = params.mission

id_prepend = 'tic'
file_append = 'tess'
cadence = 1/24/30 # 2min

directorymaker('figs')
# if forestset == 'test':
#     dir_name = 'testsetdata'
#     targets = pd.read_csv(f'S21{rot_for_file}_test.csv')
# elif forestset == 'train':
#     dir_name = 'targetdata'
#     targets = pd.read_csv(f'S21{rot_for_file}_train.csv')
# elif forestset == 'validate':
#     dir_name = 'validatedata'
#     targets = pd.read_csv(f'S21{rot_for_file}_test.csv')

dir_name = 'tesstest'
directorymaker(dir_name)

# start = time.time()

# if len(str(tid)) == 6: # this will also need a TIC version eventually
#     openstr = '000' + str(tid)
# elif len(str(tid)) == 7:
#     openstr = '00' + str(tid)
# else:
#     openstr = '0' + str(tid)

try:
    # hdu = fits.open(f'/data/shared_data/kepler/Q9/kplr{openstr}-2011177032512_llc.fits') # Q9, will need changing for TESS data and when i add other quarters
    hdu = fits.open(filename)
except FileNotFoundError:
    print(f'NO DATA: {tid}')
    sys.exit()

table = hdu[1].data
time = table['TIME']
flux = table['PDCSAP_FLUX']
hdu.close()

minfreq = 1/(time[-1] - time[0])
p_grid = np.linspace(0, time[-1], 10000)
freq = 1/p_grid

time, flux = nancleaner2d(time, flux)
time, flux = clip(time, flux, 3) #3 sigma clip
flux = lk.LightCurve(time=time, flux=flux).normalize().flux.value - 1

target = Spinner(time, flux)

ps = ts.LombScargle(time, flux).power(freq)
target.ls_one_term(freq, ps)

ps = ts.LombScargle(time, flux, nterms=2).power(freq)
target.ls_two_term(freq, ps)

lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, cadence, width=16)
target.acf(lags, acf)

fig1 = target.diagnostic_plot(heading=f'TIC {tid}')
# figsaver(fig1, f'TIC{tid}.png', '/home/isy/Documents/Work/rotation/figs')
figsaver(fig1, f'TIC{tid}_{file_append}.png')
filemaker(target, tid, 0, filename=f'{id_prepend}{tid}_{file_append}.csv', filepath=f'./{dir_name}')

print(f'{tid} done')

# end = time.time()
# print(end-start)
