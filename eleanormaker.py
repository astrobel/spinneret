import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.signal as sps
import astropy.timeseries as ts
from astropy.io import fits
import os, sys, time, argparse, eleanor
from spinneret import *

def directorymaker(dirname=f'untitled{time.time():.0f}'):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

id_prepend = 'tic'
file_append = 'eleanor'
cadence = 1/24/2 # 30min

failed = []

directorymaker('figs')

dir_name = 'kelt_eleanor'
directorymaker(dir_name)

prime_sample = np.loadtxt('prime_sectors.dat', delimiter=',')

for i in range(prime_sample.shape[0]):

    tid = int(prime_sample[i][0])
    sec = int(prime_sample[i][1])
    gtp = prime_sample[i][2]

    try:
        star = eleanor.Source(tic=tid, sector=sec, tc=True)
        data = eleanor.TargetData(star, height=15, width=15, bkg_size=31, do_psf=False, do_pca=False, regressors='corner')

        time = data.time[data.quality==0]
        flux = data.corr_flux[data.quality==0]

        os.chdir(dir_name)
        np.savetxt(f'tic{tid}_s{sec}_{file_append}_lc.dat', np.c_[time, flux], delimiter=',')
        os.chdir('..')

        minfreq = 1/(time[-1] - time[0])

        time, flux = nancleaner2d(time, flux)
        time, flux = clip(time, flux, 3) #3 sigma clip
        flux = lk.LightCurve(time=time, flux=flux).normalize().flux.value - 1

        target = Spinner(time, flux)

        freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
        target.ls_one_term(freq, ps)

        freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
        target.ls_two_term(freq, ps)

        lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, cadence, width=16)
        target.acf(lags, acf)

        fig1 = target.diagnostic_plot(heading=f'TIC {tid} s{sec} // KELT period: {gtp:.2f}d')
        figsaver(fig1, f'TIC{tid}_{file_append}.png')
        filemaker(target, tid, gtp, filename=f'{id_prepend}{tid}_s{sec}_{file_append}.csv', filepath=f'./{dir_name}')

        print(f'{tid} done')

    except eleanor.utils.SearchError:
        failed.append([tid, sec])
        print(f'{tid} failed')

np.savetxt('failed.dat', failed, delimiter=',')
