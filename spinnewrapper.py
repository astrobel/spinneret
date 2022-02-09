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
parser.add_argument('-i', '--targetid', required=True, type=int, help='KIC or TIC ID')
parser.add_argument('-s', '--rfset', required=False, default='test', type=str, choices=['test', 'train', 'validate'], help='test, train, or validate?')
parser.add_argument('-r', '--rotating', required=False, default='yes', type=str, choices=['yes', 'no', 'unknown'], help='Is the star rotating?')
parser.add_argument('-m', '--mission', required=False, default='TESS', type=str, choices=['TESS', 'TESSlike', 'Kepler'], help='TESS(like) sector or Kepler quarter?')
parser.add_argument('-n', '--number', required=True, type=int, help='TESS sector or Kepler quarter; enter 14 for TESSlike Kepler data')
# needs an on/off switch for figures, defaulting to off

params = parser.parse_args()

tid = params.targetid
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

p_r = targets['Prot'].loc[targets['KIC'].values==tid].values[0] # will need changing when i have targets with TICs

# start = time.time()

if len(str(tid)) == 6: # this will also need a TIC version eventually
    openstr = '000' + str(tid)
elif len(str(tid)) == 7:
    openstr = '00' + str(tid)
else:
    openstr = '0' + str(tid)

try:
    # hdu = fits.open(f'/data/shared_data/kepler/Q9/kplr{openstr}-2011177032512_llc.fits') # Q9, will need changing for TESS data and when i add other quarters
    hdu = fits.open(f'kplr{openstr}-2011177032512_llc.fits') # TEMP ONLY
except FileNotFoundError:
    print(f'NO DATA: {tid}')
    sys.exit()

table = hdu[1].data
time = table['TIME']
flux = table['PDCSAP_FLUX']
hdu.close()

if mission == 'TESSlike':
    time, flux = tessify(time, flux)

minfreq = 1/(time[-1] - time[0])

time, flux = nancleaner2d(time, flux)
time, flux = clip(time, flux, 3) #3 sigma clip
flux = lk.LightCurve(time=time, flux=flux).normalize().flux.value - 1

target_kep = Spinner(time, flux)

freq, ps = ts.LombScargle(time, flux).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
target_kep.ls_one_term(freq, ps)

freq, ps = ts.LombScargle(time, flux, nterms=2).autopower(nyquist_factor=1, samples_per_peak=50, minimum_frequency=minfreq)
target_kep.ls_two_term(freq, ps)

lags_raw, acf_raw, lags, acf, _x, _y = simple_acf(time, flux, cadence, width=16)
target_kep.acf(lags, acf)

fig1 = target_kep.diagnostic_plot(heading=f'KIC {tid}: Kepler Q9 // Santos 21 period = {p_r:.3f}d')
# figsaver(fig1, '/home/isy/Documents/Work/rotation/figs', f'KIC{k}_kep.png')
figsaver(fig1, f'KIC{tid}_{file_append}.png')
filemaker(target_kep, tid, p_r, filename=f'{id_prepend}{tid}_{file_append}.csv')

print(f'{tid} done')

# end = time.time()
# print(end-start)
