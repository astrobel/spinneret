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
parser.add_argument('-i', '--id', required=True, type=int, help='KIC or TIC ID')
parser.add_argument('-s', '--rfset', required=False, default='test', type=str, choices=['test', 'train', 'validate'], help='test, train, or validate?')
parser.add_argument('-r', '--rotating', required=False, default='yes', type=str, choices=['yes', 'no', 'unknown'], help='Is the star rotating?')
parser.add_argument('-m', '--mission', required=False, default='TESS', type=str, choices=['TESS', 'TESSlike', 'Kepler'], help='TESS(like) sector or Kepler quarter?')
parser.add_argument('-n', '--number', required=True, type=int, help='TESS sector or Kepler quarter; enter 14 for TESSlike Kepler data')

params = parser.parse_args()

id = params.id
forestset = params.rfset
rotating = params.rotating
mission = params.mission

def directorymaker(dirname=f'untitled{time.time():.0f}'):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

rot_for_file = ''
if rotating == 'yes':
    rot_for_file = 'r'
elif rotating == 'no':
    rot_for_file = 'n'

if mission == 'Kepler':
    id_prepend = 'kic'
    file_append = 'kepler'
    cadence = 1/24/2 # 30min
elif mission == 'TESSlike':
    id_prepend = 'kic'
    file_append = 'tess'
    cadence = 1/24/2 # 30min
elif mission == 'TESS':
    id_prepend = 'tic'
    file_append = 'tess'
    cadence = 1/24/30 # 2min

directorymaker('figs')
if forestset == 'test':
    directorymaker('testsetdata')
    targets = pd.read_csv(f'S21{rot_for_file}_test.csv')
elif forestset == 'train':
    directorymaker('targetdata')
    targets = pd.read_csv(f'S21{rot_for_file}_train.csv')
elif forestset == 'validate':
    directorymaker('validatedata')
    targets = pd.read_csv(f'S21{rot_for_file}_test.csv')

kic_r = targets['KIC']
p_r = targets['Prot']

kep_cadence = 1/24/2 # 30min cadence
tess_cadence = 1/24/30 # 2min cadence, for use later

nodata = [] # for stars with no data

# rotator sample
for i, k in enumerate(kic_r):

    # start = time.time()
    if len(str(k)) == 6:
        openstr = '000' + str(k)
    elif len(str(k)) == 7:
        openstr = '00' + str(k)
    else:
        openstr = '0' + str(k)

    try:
        hdu = fits.open(f'/data/shared_data/kepler/Q9/kplr{openstr}-2011177032512_llc.fits') # Q9, will need changing for TESS data
    except FileNotFoundError:
        print(f'NO DATA: {k}')
        nodata.append(k)
        continue

    table = hdu[1].data
    time = table['TIME']
    flux = table['PDCSAP_FLUX']
    hdu.close()

    time, flux = nancleaner2d(time, flux)
    time, flux = clip(time, flux, 3) #3 sigma clip
    flux = lk.LightCurve(time=time, flux=flux).normalize().flux.value - 1

    target_kep = Spinner(time, flux)

    freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=1, samples_per_peak=50, maximum_frequency=(1/90))
    target_kep.ls_one_term(freq, ps)

    freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=50, maximum_frequency=(1/90))
    target_kep.ls_two_term(freq, ps)

    lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, kep_cadence, width=16)
    target_kep.acf(lags, acf)

    # fig1 = target_kep.diagnostic_plot(heading=f'KIC {k}: Kepler Q9 // Santos 21 period = {p_r[i]:.3f}d')
    # figsaver(fig1, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_kep.png')
    # figsaver(fig1, f'KIC{k}_kep.png')
    filemaker(target_kep, k, p_r[i], filename=f'kic{k}_kepler.csv')

    #####

    time_tess, flux_tess = tessify(time, flux)

    target_tess = Spinner(time_tess, flux_tess)

    freq, ps = ts.LombScargle(time_tess, flux_tess).autopower(nyquist_factor=1, samples_per_peak=50, maximum_frequency=(1/27))
    target_tess.ls_one_term(freq, ps)

    freq, ps = ts.LombScargle(time_tess, flux_tess, nterms=2).autopower(nyquist_factor=1, samples_per_peak=50, maximum_frequency=(1/27))
    target_tess.ls_two_term(freq, ps)

    lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time_tess, flux_tess, kep_cadence, width=16)
    target_tess.acf(lags, acf)

    # fig2 = target_tess.diagnostic_plot(heading=f'KIC {k}: TESSify // Santos 21 period = {p_r[i]:.3f}d')
    # figsaver(fig2, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess1.png')
    # figsaver(fig2, f'KIC{k}_tess1.png')
    filemaker(target_tess, k, p_r[i], filename=f'kic{k}_tess.csv')

    #####

    # time_butter, flux_butter = tessify(time, sps.sosfilt(sos, flux))

    # target_butter = Spinner(time_butter, flux_butter)

    # freq, ps = ts.LombScargle(time_butter, flux_butter).autopower(nyquist_factor=1, samples_per_peak=30)
    # target_butter.ls_one_term(freq, ps)

    # freq, ps = ts.LombScargle(time_butter, flux_butter, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    # target_butter.ls_two_term(freq, ps)

    # lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time_butter, flux_butter, kep_cadence, width=16)
    # target_butter.acf(lags, acf)

    # fig3 = target_butter.diagnostic_plot(heading=f'KIC {k}: TESSify + 27d Butterworth filter // Santos 21 period = {p_r[i]:.3f}d')
    # # figsaver(fig3, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess2.png')
    # figsaver(fig3, f'KIC{k}_tess2.png')
    # filemaker(target_butter, k, p_r[i], filename=f'kic{k}_final.csv')

    # if i == 3:
    #     sys.exit()

    print(f'{k} done')

    # end = time.time()
    # print(end-start)
    # sys.exit()

np.savetxt('noq9.dat', nodata, delimiter='\n')
