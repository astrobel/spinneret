import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.timeseries as ts
import scipy.signal as sps
import scipy.ndimage as spn
import rubin_rotation as rr
import os

tess_ts = lk.search_lightcurve('16 Cyg B', sector=14).download()
time = tess_ts.time.value

prots = np.arange(1,31) # eventually go up to 31 for 30d periods
tau_range=(1, 3)
errs = np.linspace(0.005, 0.05, 20) # 20 different spread settings

for prot in prots:

    # remove trends longer than injected period + 2
    sos = sps.butter(3, (1/(prot+2)), 'hp', fs=24*30, output='sos')

    for err in errs:

        sin2incl = np.random.uniform(np.sin(0)**2, np.sin(np.pi/2)**2)
        incl = np.arcsin(sin2incl**.5)
        tau = np.exp(np.random.uniform(np.log(tau_range[0]*prot), np.log(tau_range[1]*prot)))

        res0, res1 = rr.mklc(time, incl=incl, tau=tau, p=prot)
        nspot, ff, amp_err = res0
        _, area_tot, dF_tot, dF_tot0 = res1
        pure_flux = dF_tot0 / np.median(dF_tot0) - 1

        flux = pure_flux + np.random.randn(len(time)) * err
        # flux_err = np.ones_like(flux)*err
        flux = sps.sosfilt(sos, flux)

        pstring = None
        if prot < 10:
            pstring = '0'+f'{prot:d}'
        else:
            pstring = str(prot)   
        syn_id = f'SYN-{pstring}-{err:.3f}'

        os.chdir('./synthetica_all')
        np.savetxt(f'{syn_id}.dat', np.c_[time, flux])

        fig, ax = plt.subplots(1)
        ax.plot(time, flux, 'k.', ms=1)
        ax.set(xlabel='time (d)', ylabel='flux')
        plt.savefig(f'{syn_id}.png')
        plt.close()

        os.chdir('..')