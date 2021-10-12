import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.signal as sps
import astropy.timeseries as ts
from astropy.io import fits
import os, sys, time
from spinneret import *

rotators = pd.read_csv('M14rotators.csv')
# control = pd.read_csv('M14nonrotators.csv')

kic_r = rotators['KIC']
p_r = rotators['Prot']
# kic_c = control['KIC']

kep_cadence = 1/24/2 # 30min cadence
tess_cadence = 1/24/30 # 2min cadence, for use later

# butterworth filter for tessify data
sos = sps.butter(3, (1/27), 'hp', fs=48, output='sos')


# rotator sample
for i, k in enumerate(kic_r):

    # start = time.time()
    if len(str(k)) == 6:
        openstr = '000' + str(k)
    elif len(str(k)) == 7:
        openstr = '00' + str(k)
    else:
        openstr = '0' + str(k)

    hdu = fits.open(f'/data/shared_data/kepler/Q9/kplr{openstr}-2011177032512_llc.fits') # Q9, will need changing for TESS data
    table = hdu[1].data
    time = table['TIME']
    flux = table['PDCSAP_FLUX']
    time, flux = nancleaner2d(time, flux)
    time, flux = clip(time, flux, 3) #3 sigma clip
    flux = flux / np.linalg.norm(flux)

    target_kep = Spinner(time, flux)

    freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=1, samples_per_peak=30)
    target_kep.ls_one_term(freq, ps)

    freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    target_kep.ls_two_term(freq, ps)

    lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, kep_cadence, width=16)
    target_kep.acf(lags, acf)

    # os.chdir('./acfs')
    # np.savetxt(f'KIC{k}_kep_acf.csv', np.c_[lags_raw, acf_raw], delimiter=',')
    # os.chdir('..')

    fig1 = target_kep.diagnostic_plot(heading=f'KIC {k}: Kepler Q9 // McQuillan 14 period = {p_r[i]:.3f}d')
    # figsaver(fig1, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_kep.png')
    figsaver(fig1, './figs', f'KIC{k}_kep.png')
    filemaker(target_kep, k, p_r[i], filepath='./targetdata', filename=f'kic{k}_kepler.csv')

    #####

    time_tess, flux_tess = tessify(time, flux)

    target_tess = Spinner(time_tess, flux_tess)

    freq, ps = ts.LombScargle(time_tess, flux_tess).autopower(nyquist_factor=1, samples_per_peak=30)
    target_tess.ls_one_term(freq, ps)

    freq, ps = ts.LombScargle(time_tess, flux_tess, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    target_tess.ls_two_term(freq, ps)

    lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time_tess, flux_tess, kep_cadence, width=16)
    target_tess.acf(lags, acf)

    # os.chdir('./acfs')
    # np.savetxt(f'KIC{k}_tess1_acf.csv', np.c_[lags_raw, acf_raw], delimiter=',')
    # os.chdir('..')
    
    fig2 = target_tess.diagnostic_plot(heading=f'KIC {k}: TESSify // McQuillan 14 period = {p_r[i]:.3f}d')
    # figsaver(fig2, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess1.png')
    figsaver(fig2, './figs', f'KIC{k}_tess1.png')
    filemaker(target_tess, k, p_r[i], filepath='./targetdata', filename=f'kic{k}_tess.csv')

    #####

    flux_butter = sps.sosfilt(sos, flux_tess)

    target_butter = Spinner(time_tess, flux_butter)

    freq, ps = ts.LombScargle(time_tess, flux_butter).autopower(nyquist_factor=1, samples_per_peak=30)
    target_butter.ls_one_term(freq, ps)

    freq, ps = ts.LombScargle(time_tess, flux_butter, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    target_butter.ls_two_term(freq, ps)

    lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time_tess, flux_butter, kep_cadence, width=16)
    target_butter.acf(lags, acf)

    # os.chdir('./acfs')
    # np.savetxt(f'KIC{k}_tess2_acf.csv', np.c_[lags_raw, acf_raw], delimiter=',')
    # os.chdir('..')

    fig3 = target_butter.diagnostic_plot(heading=f'KIC {k}: TESSify + 27d Butterworth filter // McQuillan 14 period = {p_r[i]:.3f}d')
    # figsaver(fig3, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess2.png')
    figsaver(fig3, './figs', f'KIC{k}_tess2.png')
    filemaker(target_butter, k, p_r[i], filepath='./targetdata', filename=f'kic{k}_final.csv')

    # if i == 3:
    #     sys.exit()

    print(f'{k} done')

    # end = time.time()
    # print(end-start)
    # sys.exit()