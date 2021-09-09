import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib as mpl
# import starspot as ss
import pandas as pd
import scipy.signal as sps
import astropy.timeseries as ts
# import exoplanet as ex
import os, sys, time
from tessifystarspot import *

# mpl.use('pgf')
# mpl.rc('text', usetex=True)
# # mpl.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \n \usepackage[EULERGREEK]{sansmath} \n \sansmath'
# mpl.rcParams['pgf.preamble'] = '\n'.join([r'\usepackage{helvet}', r'\usepackage[EULERGREEK]{sansmath}', r'\sansmath'])
# mpl.rcParams['axes.formatter.useoffset'] = False
# # mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# mpl.rcParams['ps.useafm'] = True
# mpl.rcParams['pdf.use14corefonts'] = True
# mpl.rcParams['font.size'] = 1

rotators = pd.read_csv('M14rotators_temp.csv')
# control = pd.read_csv('M14nonrotators.csv')

kic_r = rotators['KIC']
p_r = rotators['Prot']
# kic_c = control['KIC']

kep_cadence = 1/24/2 # 30min cadence
tess_cadence = 1/24/2 # 30min cadence

qual_kep = np.zeros(len(kic_r))
qual_tess = np.zeros(len(kic_r))
p_ls_kep = np.zeros(len(kic_r))
p_ls_kep2 = np.zeros(len(kic_r))
p_acf_kep = np.zeros(len(kic_r))
p_ls_tess = np.zeros(len(kic_r))
p_ls_tess2 = np.zeros(len(kic_r))
p_acf_tess = np.zeros(len(kic_r))
p_ls_butter = np.zeros(len(kic_r))
p_ls_butter2 = np.zeros(len(kic_r))
p_acf_butter = np.zeros(len(kic_r))

# butterworth filter for tessify data -- MOVE TO FUNCTION for testing
sos = sps.butter(3, (1/27), 'hp', fs=48, output='sos')

# check if test.csv exists
fname = 'test.csv'
if fname in os.listdir('.'):
    os.remove(fname)

f = open(fname, 'a')
f.write('KIC,McQuillan Period (d),Kepler LS Period (d),Kepler LS 2-term Period (d),Kepler ACF Period (d),TESS LS Period (d),TESS LS 2-term Period (d),TESS ACF Period (d),TESS LS Period: filtered (d),TESS LS 2-term Period: filtered (d),TESS ACF Period: filtered (d)\n')#,Kepler percentage flagged,TESS percentage flagged\n')
f.close()

# rotator sample
for i, k in enumerate(kic_r):

    # start = time.time()

    # lc_qual = lk.search_lightcurve(f'KIC {k}', quarter=9).download(quality_bitmask=0)
    # qual_kep[i] = len(np.nonzero(lc_qual.quality.value)[0]) / len(lc_qual.time)

    lc = lk.search_lightcurve(f'KIC {k}', quarter=9).download().remove_outliers()
    lc = lc.normalize() - 1 # to make butterworth filter work

    # ls_kep = lc.to_periodogram(oversample_factor=50)
    # p_ls_kep[i] = ls_kep.period_at_max_power.value 
    freq_kep, ps_kep = ts.LombScargle(lc.time, lc.flux).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_kep[i] = 1/freq_kep[np.argmax(ps_kep)].value

    freq_kep2, ps_kep2 = ts.LombScargle(lc.time, lc.flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_kep2[i] = 1/freq_kep2[np.argmax(ps_kep2)].value

    lags_kep, acf_kep, _x, _y = simple_acf(lc.time.value, lc.flux.value, kep_cadence, smooth=9, window_length=99, polyorder=3)
    try: # to remove in final pipeline, can edit simple_acf to increase lag
        p_acf_kep[i] = get_acf_period(lags_kep, acf_kep)
    except IndexError:
        p_acf_kep[i] = 0

    fig1 = diagnostic_plot(lc, freq_kep, ps_kep, freq_kep2, ps_kep2, lags_kep, acf_kep, p_ls_kep[i], p_ls_kep2[i], p_acf_kep[i], heading=f'KIC {k}: Kepler Q9 // McQuillan 14 period = {p_r[i]:.3f}')
    figsaver(fig1, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_kep.png')

    #####

    lc_tess = tessify(lc)#, start_modifier=1000)
    # qual_tess[i] = len(np.nonzero(lc_tess.quality)) / len(lc_tess.time)

    # ls_tess = lc_tess.to_periodogram(oversample_factor=50)
    # p_ls_tess[i] = ls_tess.period_at_max_power.value 
    freq_tess, ps_tess = ts.LombScargle(lc_tess.time, lc_tess.flux).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_tess[i] = 1/freq_tess[np.argmax(ps_tess)].value

    freq_tess2, ps_tess2 = ts.LombScargle(lc.time, lc.flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_tess2[i] = 1/freq_tess2[np.argmax(ps_tess2)].value

    lags_tess, acf_tess, _x, _y = simple_acf(lc_tess.time.value, lc_tess.flux.value, tess_cadence, smooth=9, window_length=99, polyorder=3)
    try:
        p_acf_tess[i] = get_acf_period(lags_tess, acf_tess)
    except IndexError:
        p_acf_tess[i] = 0

    fig2 = diagnostic_plot(lc_tess, freq_tess, ps_tess, freq_tess2, ps_tess2, lags_tess, acf_tess, p_ls_tess[i], p_ls_tess2[i], p_acf_tess[i], heading=f'KIC {k}: TESSify // McQuillan 14 period = {p_r[i]:.3f}')
    figsaver(fig2, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess1.png')

    #####

    newflux = sps.sosfilt(sos, lc.flux.value)
    lc.flux = newflux

    lc_butter = tessify(lc)#, start_modifier=1000)

    # ls_butter = lc_butter.to_periodogram(oversample_factor=50)
    # p_ls_butter[i] = ls_butter.period_at_max_power.value
    freq_butter, ps_butter = ts.LombScargle(lc_butter.time, lc_butter.flux).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_butter[i] = 1/freq_butter[np.argmax(ps_butter)].value

    freq_butter2, ps_butter2 = ts.LombScargle(lc.time, lc.flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=30)
    p_ls_butter2[i] = 1/freq_butter2[np.argmax(ps_butter2)].value

    lags_butter, acf_butter, _x, _y = simple_acf(lc_butter.time.value, lc_butter.flux.value, tess_cadence, smooth=9, window_length=99, polyorder=3)
    try:
        p_acf_butter[i] = get_acf_period(lags_butter, acf_butter)
    except IndexError:
        p_acf_butter[i] = 0

    fig3 = diagnostic_plot(lc_butter, freq_butter, ps_butter, freq_butter2, ps_butter2, lags_butter, acf_butter, p_ls_butter[i], p_ls_butter2[i], p_acf_butter[i], heading=f'KIC {k}: TESSify + 27d Butterworth filter // McQuillan 14 period = {p_r[i]:.3f}')
    figsaver(fig3, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_tess2.png')

    f = open(fname, 'a')
    f.write(f'{k},{p_r[i]},{p_ls_kep[i]},{p_ls_kep2[i]},{p_acf_kep[i]},{p_ls_tess[i]},{p_ls_tess2[i]},{p_acf_tess[i]},{p_ls_butter[i]},{p_ls_butter2[i]},{p_acf_butter[i]}') #,{qual_kep[i]},{qual_tess[i]}\n')
    f.close()

    # end = time.time()

    # print(end-start)
    # sys.exit()

    # if i == 0:
    #     break

    # print(f'{k} done')

# dict_r = {'McQuillan Period (d)':p_r, 'Kepler LS Period (d)':ls_kep, 'Kepler ACF Period (d)':acf_kep, 'TESS LS Period (d)':ls_tess, 'TESS ACF Period (d)':acf_tess, 'TESS LS Period: filtered (d)':ls_butter, 'TESS ACF Period: filtered (d)':acf_butter, 'Kepler percentage flagged':qual_kep, 'TESS percentage flagged':qual_tess}
# out_r = pd.DataFrame(data=dict_r)
# out_r.to_csv('test.csv')

# control sample
# for i, k in enumerate(kic_c):
#     try:
#         lc = lk.search_lightcurve(f'KIC {k}', quarter=9).download().remove_outliers()
#     except:
#         pass

# fig, ax = plt.subplots(2,1)

# ax[0].scatter(ls_kep, p_r)
# ax[0].set(xlabel='LS kep', ylabel='M14 Prot')

# ax[1].scatter(ls_tess, p_r)
# ax[1].set(xlabel='LS TESS', ylabel='M14 Prot')

# plt.tight_layout()
# plt.savefig('test1.png')