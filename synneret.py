import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy.signal as sps
import scipy.ndimage as spn
import astropy.timeseries as ts
import os, sys, time
from tessifystarspot import *

# change as per synthetica.py
prots = np.arange(1,11)
errs = np.linspace(0.005, 0.05, 10) 

tess_cadence = 1/24/30 # 2min cadence

# check if syn_test.csv exists
fname = 'syn_test.csv'
if fname in os.listdir('.'):
    os.remove(fname)

f = open(fname, 'a')
f.write('SYN ID,Injected Period (d),LS Period (d),LS Period RMS,LS 2-term Period (d),LS 2-term Period RMS,ACF Period (d),ACF Period RMS\n')
f.close()

# remove trends longer than ??
# sos = sps.butter(3, (1/30), 'hp', fs=24*30, output='sos')

for prot in prots:

    for err in errs:

        pstring = None
        if prot < 10:
            pstring = '0'+f'{prot:d}'
        else:
            pstring = str(prot)   
        syn_id = f'SYN-{pstring}-{err:.3f}'

        lc = np.loadtxt(f'./synthetica/{syn_id}.dat', delimiter=' ')
        time = lc[:,0]
        flux = lc[:,1]

        # flux = sps.sosfilt(sos, flux)

        # start = time.time()

        target = Spinner(time, flux)

        freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=0.5, samples_per_peak=30)
        target.ls_one_term(freq, ps)

        freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=0.5, samples_per_peak=30)
        target.ls_two_term(freq, ps)

        lags, acf, _x, _y = simple_acf(target.time, target.flux, tess_cadence, smooth=9, window_length=56)
        target.acf(lags, acf)

        fig1 = target.diagnostic_plot(heading=syn_id)
        figsaver(fig1, '/home/isy/Documents/Work/rotation/synthetica', f'{syn_id}_diagnostic.png')

        f = open(fname, 'a')
        f.write(f'{syn_id},{prot},{target.p_ls1},{target.rms_ls1},{target.p_ls2},{target.rms_ls2},{target.p_acf},{target.rms_acf}\n')
        f.close()

        print(f'{syn_id} done')

        # end = time.time()
        # print(end-start)
        # sys.exit()

# for testing
# prot = 3
# err = 0.035

# pstring = None
# if prot < 10:
#     pstring = '0'+f'{prot:d}'
# else:
#     pstring = str(prot)   
# syn_id = f'SYN-{pstring}-{err:.3f}'

# lc = np.loadtxt(f'./synthetica/{syn_id}.dat', delimiter=' ')
# time = lc[:,0]
# flux = lc[:,1]

# flux = sps.sosfilt(sos, flux)

# # start = time.time()

# target = Spinner(time, flux)

# freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=0.5, samples_per_peak=30)
# target.ls_one_term(freq, ps)

# freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=0.5, samples_per_peak=30)
# target.ls_two_term(freq, ps)

# lags, acf, _x, _y = simple_acf(target.time, target.flux, tess_cadence, smooth=9, window_length=56)
# target.acf(lags, acf)

# # np.savetxt(f'SYN-{pstring}-{err:.3f}.csv', np.c_[lags, acf], delimiter=',')

# fig1 = target.diagnostic_plot(heading=syn_id)
# figsaver(fig1, '/home/isy/Documents/Work/rotation/synthetica', f'{syn_id}_diagnostic.png')

# fig2, ax = plt.subplots(1)
# plt.plot(lags, acf)
# plt.show()