import pandas as pd
import numpy as np
import lightkurve as lk
import scipy.interpolate as spi
import scipy.signal as sps
import scipy.ndimage as spn
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.timeseries as ts
from astropy import units as u
import os, sys

C = ['#ff6897','#fe6798','#fd6799','#fc679a','#fb669b','#fa669d','#f9669e','#f8669f','#f765a0','#f665a1','#f565a2','#f464a4','#f364a5','#f264a6','#f163a7','#f063a8','#ef63a9','#ee63ab','#ed62ac','#ec62ad','#eb62ae','#ea61af','#e961b1','#e861b2','#e761b3','#e660b4','#e560b5','#e460b6','#e35fb8','#e25fb9','#e15fba','#e05fbb','#df5ebc','#de5ebd','#dd5ebf','#dc5dc0','#db5dc1','#da5dc2','#d95cc3','#d85cc5','#d75cc6','#d65cc7','#d55bc8','#d45bc9','#d35bca','#d25acc','#d15acd','#d05ace','#cf5acf','#ce59d0','#cd59d1','#cc59d3','#cb58d4','#ca58d5','#c958d6','#c858d7','#c757d9','#c657da','#c557db','#c456dc','#c356dd','#c256de','#c156e0','#c055e1','#bf55e2','#bf55e2','#be55e2','#bd55e3','#bc56e3','#bc56e4','#bb56e4','#ba56e4','#b956e5','#b956e5','#b856e5','#b757e6','#b657e6','#b657e6','#b557e7','#b457e7','#b357e8','#b258e8','#b258e8','#b158e9','#b058e9','#af58e9','#af58ea','#ae58ea','#ad59ea','#ac59eb','#ac59eb','#ab59ec','#aa59ec','#a959ec','#a959ed','#a85aed','#a75aed','#a65aee','#a55aee','#a55aee','#a45aef','#a35aef','#a25bf0','#a25bf0','#a15bf0','#a05bf1','#9f5bf1','#9f5bf1','#9e5bf2','#9d5cf2','#9c5cf2','#9c5cf3','#9b5cf3','#9a5cf4','#995cf4','#995cf4','#985df5','#975df5','#965df5','#955df6','#955df6','#945df6','#935df7','#925ef7','#925ef8','#915ef8','#905ef8','#8f5ef9','#8e5ff8','#8d61f7','#8c63f6','#8b65f5','#8a66f4','#8968f3','#886af2','#876cf1','#856ef0','#846fee','#8371ed','#8273ec','#8175eb','#8076ea','#7f78e9','#7e7ae8','#7d7ce7','#7b7ee6','#7a7fe5','#7981e4','#7883e2','#7785e1','#7687e0','#7588df','#748ade','#738cdd','#718edc','#7090db','#6f91da','#6e93d9','#6d95d8','#6c97d6','#6b99d5','#6a9ad4','#689cd3','#679ed2','#66a0d1','#65a2d0','#64a3cf','#63a5ce','#62a7cd','#61a9cc','#60abca','#5eacc9','#5daec8','#5cb0c7','#5bb2c6','#5ab3c5','#59b5c4','#58b7c3','#57b9c2','#56bbc1','#54bcc0','#53bebe','#52c0bd','#51c2bc','#50c4bb','#4fc5ba','#4ec7b9','#4dc9b8','#4bcbb7','#4acdb6','#49ceb5','#48d0b4','#49d1b2','#4ad1b0','#4bd2ae','#4cd2ad','#4dd2ab','#4ed3a9','#4fd3a7','#50d4a5','#51d4a3','#52d4a2','#53d5a0','#54d59e','#55d59c','#56d69a','#57d699','#58d797','#59d795','#5ad793','#5bd891','#5cd890','#5dd88e','#5ed98c','#5fd98a','#60da88','#61da87','#62da85','#63db83','#64db81','#65db7f','#67dc7e','#68dc7c','#69dd7a','#6add78','#6bdd76','#6cde75','#6dde73','#6ede71','#6fdf6f','#70df6d','#71e06b','#72e06a','#73e068','#74e166','#75e164','#76e162','#77e261','#78e25f','#79e35d','#7ae35b','#7be359','#7ce458','#7de456','#7ee454','#7fe552','#80e550','#81e64f','#82e64d','#83e64b','#84e749','#85e747','#86e746','#87e844','#89e842','#8ae940']
lccmap = mpl.colors.ListedColormap(C)


class Spinner:

    def __init__(self, time, flux):
        self.time = time
        self.flux = flux
        self.rvar = rvar(flux)
        self.cdpp = lk.LightCurve(time=time * u.d, flux=(flux+1) * u.d).estimate_cdpp(transit_duration=4).value

    def ls_one_term(self, freq, ps):
        self.freq1 = freq
        self.ps1 = ps
        self.ps1_med = np.median(ps)
        xp, yp = find_all_peaks(freq, ps)
        self.p_ls1a = 1/xp[0]
        self.p_ls1b = 1/xp[1]
        self.p_ls1c = 1/xp[2]
        self.a_ls1a = yp[0]
        self.a_ls1b = yp[1]
        self.a_ls1c = yp[2]

        self.flux_ls1a_fold, self.model_ls1a = model_for_stats(self.time, self.flux, self.p_ls1a)
        self.flux_ls1b_fold, self.model_ls1b = model_for_stats(self.time, self.flux, self.p_ls1b)
        self.flux_ls1c_fold, self.model_ls1c = model_for_stats(self.time, self.flux, self.p_ls1c)
        self.rms_ls1a = rms(self.model_ls1a, self.flux_ls1a_fold)
        self.rms_ls1b = rms(self.model_ls1b, self.flux_ls1b_fold)
        self.rms_ls1c = rms(self.model_ls1c, self.flux_ls1c_fold)
        self.mad_ls1a = mad(self.model_ls1a, self.flux_ls1a_fold)
        self.mad_ls1b = mad(self.model_ls1b, self.flux_ls1b_fold)
        self.mad_ls1c = mad(self.model_ls1c, self.flux_ls1c_fold)

        self.time_ls1a_plot, self.flux_ls1a_plot, self.time_orig_ls1a_plot, self.model_ls1a_plot = model_for_plotting(self.time, self.flux, self.p_ls1a)
        self.time_ls1b_plot, self.flux_ls1b_plot, self.time_orig_ls1b_plot, self.model_ls1b_plot = model_for_plotting(self.time, self.flux, self.p_ls1b)
        self.time_ls1c_plot, self.flux_ls1c_plot, self.time_orig_ls1c_plot, self.model_ls1c_plot = model_for_plotting(self.time, self.flux, self.p_ls1c)

    def ls_two_term(self, freq, ps):
        self.freq2 = freq
        self.ps2 = ps
        self.ps2_med = np.median(ps)
        xp, yp = find_all_peaks(freq, ps)
        self.p_ls2a = 1/xp[0]
        self.p_ls2b = 1/xp[1]
        self.p_ls2c = 1/xp[2]
        self.a_ls2a = yp[0]
        self.a_ls2b = yp[1]
        self.a_ls2c = yp[2]

        self.flux_ls2a_fold, self.model_ls2a = model_for_stats(self.time, self.flux, self.p_ls2a)
        self.flux_ls2b_fold, self.model_ls2b = model_for_stats(self.time, self.flux, self.p_ls2b)
        self.flux_ls2c_fold, self.model_ls2c = model_for_stats(self.time, self.flux, self.p_ls2c)
        self.rms_ls2a = rms(self.model_ls2a, self.flux_ls2a_fold)
        self.rms_ls2b = rms(self.model_ls2b, self.flux_ls2b_fold)
        self.rms_ls2c = rms(self.model_ls2c, self.flux_ls2c_fold)
        self.mad_ls2a = mad(self.model_ls2a, self.flux_ls2a_fold)
        self.mad_ls2b = mad(self.model_ls2b, self.flux_ls2b_fold)
        self.mad_ls2c = mad(self.model_ls2c, self.flux_ls2c_fold)

        self.time_ls2a_plot, self.flux_ls2a_plot, self.time_orig_ls2a_plot, self.model_ls2a_plot = model_for_plotting(self.time, self.flux, self.p_ls2a)
        self.time_ls2b_plot, self.flux_ls2b_plot, self.time_orig_ls2b_plot, self.model_ls2b_plot = model_for_plotting(self.time, self.flux, self.p_ls2b)
        self.time_ls2c_plot, self.flux_ls2c_plot, self.time_orig_ls2c_plot, self.model_ls2c_plot = model_for_plotting(self.time, self.flux, self.p_ls2c)

    def acf(self, lags, acff):
        self.lags = lags
        self.acf = acff
        self.p_acfa, self.p_acfb, self.p_acfc, self.a_acfa, self.a_acfb, self.a_acfc = get_acf_period(lags, acff)

        if self.p_acfa != None:
            self.flux_acfa_fold, self.model_acfa = model_for_stats(self.time, self.flux, self.p_acfa)
            self.rms_acfa = rms(self.model_acfa, self.flux_acfa_fold)
            self.mad_acfa = mad(self.model_acfa, self.flux_acfa_fold)
            self.time_acfa_plot, self.flux_acfa_plot, self.time_orig_acfa_plot, self.model_acfa_plot = model_for_plotting(self.time, self.flux, self.p_acfa)
        else:
            self.flux_acfa_fold = None
            self.model_acfa = None
            self.rms_acfa = None
            self.mad_acfa = None
            self.time_acfa_plot = None
            self.flux_acfa_plot = None
            self.time_orig_acfa_plot = None
            self.model_acfa_plot = None
        if self.p_acfb != None:
            self.flux_acfb_fold, self.model_acfb = model_for_stats(self.time, self.flux, self.p_acfb)
            self.rms_acfb = rms(self.model_acfb, self.flux_acfb_fold)
            self.mad_acfb = mad(self.model_acfb, self.flux_acfb_fold)
            self.time_acfb_plot, self.flux_acfb_plot, self.time_orig_acfb_plot, self.model_acfb_plot = model_for_plotting(self.time, self.flux, self.p_acfb)
        else:
            self.flux_acfb_fold = None
            self.model_acfb = None
            self.rms_acfb = None
            self.mad_acfb = None
            self.time_acfb_plot = None
            self.flux_acfb_plot = None
            self.time_orig_acfb_plot = None
            self.model_acfb_plot = None
        if self.p_acfc != None:
            self.flux_acfc_fold, self.model_acfc = model_for_stats(self.time, self.flux, self.p_acfc)
            self.rms_acfc = rms(self.model_acfc, self.flux_acfc_fold)
            self.mad_acfc = mad(self.model_acfc, self.flux_acfc_fold)
            self.time_acfc_plot, self.flux_acfc_plot, self.time_orig_acfb_plot, self.model_acfc_plot = model_for_plotting(self.time, self.flux, self.p_acfc)
        else:
            self.flux_acfc_fold = None
            self.model_acfc = None
            self.rms_acfc = None
            self.mad_acfc = None
            self.time_acfc_plot = None
            self.flux_acfc_plot = None
            self.time_orig_acfc_plot = None
            self.model_acfc_plot = None

    def diagnostic_plot(self, heading=' '):

        # error lines
        # line_ls1 = np.poly1d([0.00793679, 0.01704705])
        # line_ls2 = np.poly1d([-0.00324059,  0.46041767])
        # line_acf = np.poly1d([0.0082101, 0.0300939])

        mosaic = """
            AA
            BC
            DE
            FG
        """

        fig = plt.figure(constrained_layout=True)
        ax = fig.subplot_mosaic(mosaic)

        if self.p_acfa != None:
            xmax = max(self.p_ls1a, self.p_ls2a, self.p_acfa)
        else:
            xmax = max(self.p_ls1a, self.p_ls2a)

        ax['A'].scatter(self.time, self.flux, c=self.time, s=3, cmap=lccmap)# '.', c='#4d0e02', ms='3')
        ax['A'].set(xlabel='time (d)', ylabel='normalized flux', title=heading, xlim=(min(self.time), max(self.time)))

        # 1-term LS
        # ax['B'].axvspan(self.p_ls1-line_ls1(self.p_ls1), self.p_ls1+line_ls1(self.p_ls1), color='#86fa20')
        ax['B'].axvline(self.p_ls1a, c='#86fa20', ls='-', lw=10)
        ax['B'].axvline(self.p_ls1b, c='#20d4fa', ls='-', lw=6, alpha=0.75)
        ax['B'].axvline(self.p_ls1c, c='#fa20c2', ls='-', lw=3, alpha=0.5)
        ax['B'].plot(1/self.freq1, self.ps1, c='#4d0e02')
        # ax['B'].axvline(self.p_ls1, c='#33ffbe', ls='--')
        ax['B'].set(xlabel='period (d)', ylabel='power', xlim=(0,xmax+10)) # xscale='log', xlim=(min(1/self.freq1), max(1/self.freq1))

        ax['C'].scatter(self.time_ls1a_plot, self.flux_ls1a_plot, c=self.time_orig_ls1a_plot, s=3, cmap=lccmap)#, '.', c='#4ef5c0', ms='3')
        ax['C'].plot(self.time_ls1a_plot, self.model_ls1a_plot, c='#4d0e02', alpha=0.8, lw=7)
        ax['C'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'LS period: {self.p_ls1a:.3f}d', xlim=(min(self.time_ls1a_plot), max(self.time_ls1a_plot)))

        # 2-term LS
        # ax['D'].axvspan(self.p_ls2-line_ls2(self.p_ls2), self.p_ls2+line_ls2(self.p_ls2), color='#86fa20')
        ax['D'].axvline(self.p_ls2a, c='#86fa20', ls='-', lw=10)
        ax['D'].axvline(self.p_ls2b, c='#20d4fa', ls='-', lw=6, alpha=0.75)
        ax['D'].axvline(self.p_ls2c, c='#fa20c2', ls='-', lw=3, alpha=0.5)
        ax['D'].plot(1/self.freq2, self.ps2, c='#4d0e02')
        # ax['D'].axvline(self.p_ls2, c='#33ffbe', ls='--')
        ax['D'].set(xlabel='period (d)', ylabel='power', xlim=(0,xmax+10))

        ax['E'].scatter(self.time_ls2a_plot, self.flux_ls2a_plot, c=self.time_orig_ls2a_plot, s=3, cmap=lccmap)#, '.', c='#4ef5c0', ms='3')
        ax['E'].plot(self.time_ls2a_plot, self.model_ls2a_plot, c='#4d0e02', alpha=0.8, lw=7)
        ax['E'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'LS period: {self.p_ls2a:.3f}d', xlim=(min(self.time_ls2a_plot), max(self.time_ls2a_plot)))

        # ACF
        # ax['F'].axvspan(self.p_acf-line_acf(self.p_acf), self.p_acf+line_acf(self.p_acf), color='#86fa20')
        if self.p_acfa != None:
            ax['F'].axvline(self.p_acfa, c='#86fa20', ls='-', lw=10)
            ax['G'].scatter(self.time_acfa_plot, self.flux_acfa_plot, c=self.time_orig_acfa_plot, s=3, cmap=lccmap)#, '.', c='#ff549b', ms='3')
            ax['G'].plot(self.time_acfa_plot, self.model_acfa_plot, c='#4d0e02', alpha=0.8, lw=7)
            ax['G'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'ACF period: {self.p_acfa:.3f}d', xlim=(min(self.time_acfa_plot), max(self.time_acfa_plot)))
        if self.p_acfb != None:
            ax['F'].axvline(self.p_acfb, c='#20d4fa', ls='-', lw=6, alpha=0.75)
        if self.p_acfc != None:
            ax['F'].axvline(self.p_acfc, c='#fa20c2', ls='-', lw=3, alpha=0.5)

        ax['F'].plot(self.lags, self.acf, c='#4d0e02')
        # ax['F'].axvline(self.p_acf, c='#33ffbe', ls='--')
        ax['F'].set(xlim=(0,xmax+10), xlabel='lags', ylabel='ACF')

        fig.set_size_inches(9,12)

        return fig


def model_for_plotting(t, f, period, terms=1):

    fold = lk.LightCurve(time=t * u.d, flux=f * u.d).fold(period)
    model = ts.LombScargle(fold.time.value, fold.flux.value, nterms=terms).model(fold.time.value, 1/period)

    return fold.time.value, fold.flux.value, fold.time_original.value, model


def model_for_stats(t, f, period, terms=1):

    fold = lk.LightCurve(time=t * u.d, flux=f * u.d).fold(period).bin(time_bin_size=period/100).remove_nans()
    model = ts.LombScargle(fold.time.value, fold.flux.value, nterms=terms).model(fold.time.value, 1/period)

    return fold.flux.value, model


def rms(model, measured):
    """
    Function to find the RMS between two curves

    Args:
        model (Numpy array): model flux measurements
        measured (Numpy array): light curve flux

    Returns:
        rms (float): RMS error between model and LC
    """

    return np.sqrt(sum(np.power(model - measured, 2))/len(measured))


def mad(model, measured):
    """
    Function to find the MAD between two curves

    Args:
        model (Numpy array): model flux measurements
        measured (Numpy array): light curve flux

    Returns:
        mad (float): MAD error between model and LC
    """

    return np.median(np.abs(model - measured))


def rvar(flux):
    """
    Function to compute R_var for a given light curve

    Args:
        flux (Numpy array): a (folded) light curve

    Returns:
        rvar (float): difference between 95th and 5th percentile of scatter
    """

    return np.percentile(flux, 95) - np.percentile(flux, 5)


def clip(time, flux, bounds):
    """
    Quick function for outlier clipping

    Args:
        time (Numpy array): time
        flux (Numpy array): PDCSAP flux
        bounds (float): number of standard deviations to clip

    Returns:
        timenew (Numpy array): clipped time
        fluxnew (Numpy array): clipped flux
    """

    sigma = np.std(flux)
    avg = np.mean(flux)

    outliers = [int(index) for f, index in zip(flux, range(len(flux))) if f > avg + sigma*bounds or f < avg - sigma*bounds]

    fluxnew = np.delete(flux, outliers)
    timenew = np.delete(time, outliers)

    return timenew, fluxnew


def nancleaner2d(time, flux):

   blend = np.array([time, flux])
   blend = np.transpose(blend)
   blend2 = np.ma.compress_rows(np.ma.fix_invalid(blend))

   timenew = blend2[:,0]
   fluxnew = blend2[:,1]

   return timenew, fluxnew


def tessify(time, flux, sector=14, start_modifier=0):
    """
    Takes a single quarter of Kepler time series data and trims it to match
    the length and shape of a TESS time series.

    Args:
        time (Numpy array): single Kepler quarter time series
        flux (Numpy array): single Kepler quarter flux series
        sector (Optional[int]): TESS sector to mimic
        start_modifier (Optional[int]): First cadence to include in the
            TESSified time series.

    Returns:
        timenew (Numpy array): single TESS-like sector time series
        fluxnew (Numpy array): single TESS-like sector flux series
    """

    # get tess orbit timing
    tess_orbits = pd.read_csv('https://tess.mit.edu/wp-content/uploads/orbit_times_20201013_1338.csv', skiprows=5)
    sectors = tess_orbits['Sector']
    starts = tess_orbits['Start TJD']
    ends = tess_orbits['End TJD']

    # get cadence numbers for stop and start points
    try:
        correction = starts[sectors==sector].iloc[0] - time[0+start_modifier]
        start1 = time[0+start_modifier]
        end1 = ends[sectors==sector].iloc[0] - correction
        start2 = starts[sectors==sector].iloc[1] - correction
        end2 = ends[sectors==sector].iloc[1] - correction
    except IndexError:
        raise IndexError('Data selected is out of range. Try using a lower start_modifier')

    timenew = np.r_[time[(time >= start1) & (time <= end1)], time[(time >= start2) & (time <= end2)]]
    fluxnew = np.r_[flux[(time >= start1) & (time <= end1)], flux[(time >= start2) & (time <= end2)]]

    return timenew, fluxnew


def interp(x_gaps, y_gaps, interval, interp_style="zero"):
    """
    *** FROM STARSPOT ***
    Interpolate the light curve

    Args:
        x_gaps (array): The time array with gaps.
        y_gaps (array): The flux array with gaps.
        interval (float): The grid to interpolate to.
        interp_style (string): The type of interpolation, e.g. "zero" or
            "linear". The default is "zero".

    Returns:
        time (array): The interpolated time array.
        flux (array): The interpolated flux array.
    """
    f = spi.interp1d(x_gaps, y_gaps, kind=interp_style)
    x = np.arange(x_gaps[0], x_gaps[-1], interval)
    return x, f(x)


def dan_acf(x, axis=0, fast=False):
    """
    *** FROM STARSPOT ***
    Estimate the autocorrelation function of a time series using the FFT.

    Args:
        x (array): The time series. If multidimensional, set the time axis
            using the ``axis`` keyword argument and the function will be
            computed for every other axis.
        axis (Optional[int]): The time axis of ``x``. Assumed to be the first
            axis if not specified.
        fast (Optional[bool]): If ``True``, only use the largest ``2^n``
            entries for efficiency. (default: False)

    Returns:
        acf (array): The acf array.
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[m]


def simple_acf(x_gaps, y_gaps, interval, smooth=9, width=16, window_length=99,
               polyorder=3, interp_style="zero"):
    """
    *** FROM STARSPOT ***
    Compute an autocorrelation function and a period.

    Applies interpolation, smoothing and peak detection to estimate a
    rotation period.

    Args:
        x_gaps (array): The time array.
        y_gaps (array): The flux array.
        interval (Optional[float]): The time interval between successive
            observations. The default is Kepler cadence.
        smooth (Optional[float]): The smoothing timescale.
        width (Optional[int]): Width for Gaussian smoothing kernel.
        window_length (Optional[float]): The filter window length.
        polyorder (Optional[float]): The polynomial order of the filter.
        interp_style (string): The type of interpolation, e.g. "zero" or
            "linear". The default is "zero".

    Returns:
        lags (array): The array of lag times in days.
        acf (array): The autocorrelation function.
        period (float): The period estimated from the highest peak in the ACF.
    """

    # First of all: interpolate to an evenly spaced grid
    x, y = interp(x_gaps, y_gaps, interval, interp_style=interp_style)

    # fit and subtract straight line
    AT = np.vstack((x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    m, b = np.linalg.solve(ATA, np.dot(AT, y))
    y -= m*x + b

    # perform acf
    acf = dan_acf(y)

    # create 'lags' array
    lags = np.arange(len(acf))*interval

    # ditch the first point
    acf, lags = acf[1:], lags[1:]

    N = len(acf)
    double_acf, double_lags = [np.zeros((2*N)) for i in range(2)]
    double_acf[:N], double_lags[:N] = acf[::-1], -lags[::-1]
    double_acf[N:], double_lags[N:] = acf, lags
    acf, lags = double_acf, double_lags
    lags1 = lags

    # Smooth the data with a Savitsky-Golay filter.
    # acf_smooth = sps.savgol_filter(acf, window_length, polyorder)
    acf_smooth = spn.gaussian_filter(acf, sigma=window_length)

    # just use the second bit (no reflection)
    acf_smooth, lags = acf_smooth[N:], lags[N:]

    return lags1, acf, lags, acf_smooth, x, y


def get_peak_statistics(x, y, sort_by="height"):
    """
    *** FROM STARSPOT ***
    Get the positions and height of peaks in an array.

    Args:
        x (array): the x array (e.g. period or lag).
        y (array): the y array (e.g. power or ACF).
        sort_by (str): The way to sort the peak array. if "height", sort peaks
            in order of height, if "position", sort peaks in order of
            x-position.

    Returns:
        x_peaks (array): the peak x-positions in descending height order, or
            ascending x-position order.
        y_peaks (array): the peak heights in descending height order, or
            ascending x-position order.
    """

    # Array of peak indices
    peaks = np.array([i for i in range(1, len(y)-1) if y[i-1] <
                      y[i] and y[i+1] < y[i]])

    # extract peak values
    try:
        x_peaks = x[peaks]
        y_peaks = y[peaks]
    except IndexError:
        x_peaks = [None]
        y_peaks = [None]
        return x_peaks, y_peaks

    # sort by height
    if sort_by == "height":
        inds = np.argsort(y_peaks)
        x_peaks, y_peaks = x_peaks[inds][::-1], y_peaks[inds][::-1]

    # sort by position
    elif sort_by == "position":
        inds = np.argsort(x_peaks)
        x_peaks, y_peaks = x_peaks[inds], y_peaks[inds]

    return x_peaks, y_peaks


def get_acf_period(lags, acf, cutoff=0):
    """
    *** FROM STARSPOT ***
    A quick function to get ACF period, adapted from Starspot

    Args:
        lags (Numpy array): lags
        acf (Numpy array): ACF
        cutoff (Optional[float]): The number of days to cut off at the
            beginning.

    Returns:
        acf_period (float): rotation period in days
    """

    # find all the peaks
    m = lags > cutoff
    xpeaks, ypeaks = get_peak_statistics(lags[m], acf[m],
                                         sort_by="height")
    if len(xpeaks) == 1:
        xpeaks = np.r_[xpeaks, None, None]
        ypeaks = np.r_[ypeaks, None, None]
    elif len(xpeaks) == 2:
        xpeaks = np.r_[xpeaks, None]
        ypeaks = np.r_[ypeaks, None]

    return xpeaks[0], xpeaks[1], xpeaks[2], ypeaks[0], ypeaks[1], ypeaks[2]


def find_all_peaks(x, y):
    """
    *** FROM STARSPOT ***
    """
    # Array of peak indices
    peaks = np.array([i for i in range(1, len(y)-1) if y[i-1] <
                      y[i] and y[i+1] < y[i]])

    # extract peak values
    x_peaks = x[peaks]
    y_peaks = y[peaks]

    # sort by height
    inds = np.argsort(y_peaks)
    x_peaks, y_peaks = x_peaks[inds][::-1], y_peaks[inds][::-1]

    return x_peaks, y_peaks


def find_peaks_at_integers(period, peak_positions):
    """
    *** FROM STARSPOT ***
    A function to identify the positions of the peaks that lie within
    10% of an integer multiple of the period.

    Args:
        period (float): The best estimate of the rotation period
        peak_positions (array): an array of the times of peaks in an ACF

    Returns:
        good_peaks (array): The positions of peaks near integer multiples.
        errs (array): an array containing the differences between the peak positions
            and the integer multiple of the period
    """

    # Calculate the modulus
    mods = peak_positions % period

    # Shift mods to make more Gaussian
    corrected_mods = mods*1
    corrected_mods[corrected_mods > .5*period] -= period

    # Find peaks within 10% of an integer multiple of the period
    good = abs(corrected_mods) < .1*period
    good_peaks = peak_positions[good]
    errs = abs(corrected_mods)[good]

    return good_peaks, errs


def fit_line_to_good_peaks(good_peaks, errs):
    """
    *** FROM STARSPOT ***
    A function that fits a line to the peak positions and pulls out a period and an uncertainty.

    Args:
        good_peaks (array): The positions of peaks within 10% of an integer multiple in an ACF
        errs (array): an array containing the differences between the peak positions
            and the integer multiple of the period

    Returns:
        The period and uncertainty
    """
    errs[errs == 0] += .001
    x = np.arange(len(good_peaks), dtype="float64")
    y = good_peaks
    w, sig = fit_line(x, y, errs)
    return w[1], sig[1]


def fit_line(x, y, yerr):
    """
    *** FROM STARSPOT ***
    """
    AT = np.vstack((np.ones(len(x)), x))
    C = np.eye(len(x))*yerr
    CA = np.linalg.solve(C, AT.T)
    Cy = np.linalg.solve(C, y)
    ATCA = np.dot(AT, CA)
    ATCy = np.dot(AT, Cy)
    w = np.linalg.solve(ATCA, ATCy)

    cov = np.linalg.inv(ATCA)
    sig = np.sqrt(np.diag(cov))
    return w, sig


def figsaver(fig, filename, filepath='./figs'):

    cwd = os.getcwd()
    os.chdir(filepath)
    fig.savefig(filename)
    os.chdir(cwd)
    plt.close(fig)


def filemaker(spinner, kic, p_r, filename=None, filepath='./targetdata'):
    """
    Args:
        spinner: Spinner object
    """

    if filename == None:
        filename = f'{kic}.csv'

    output_dict = {'KIC':kic,'Santos Period (d)':p_r,
        'LS Period 1st peak (d)':spinner.p_ls1a,'LS Period 2nd peak (d)':spinner.p_ls1b,'LS Period 3rd peak (d)':spinner.p_ls1c,
        'LS Period 1st amplitude':spinner.a_ls1a,'LS Period 2nd amplitude':spinner.a_ls1b,'LS Period 3rd amplitude':spinner.a_ls1c,
        'LS Period 1st RMS':spinner.rms_ls1a,'LS Period 1st MAD':spinner.mad_ls1a,
        'LS Period 2nd RMS':spinner.rms_ls1b,'LS Period 2nd MAD':spinner.mad_ls1b,
        'LS Period 3rd RMS':spinner.rms_ls1c,'LS Period 3rd MAD':spinner.mad_ls1c,
        'LS 2-term Period 1st peak (d)':spinner.p_ls2a,'LS 2-term Period 2nd peak (d)':spinner.p_ls2b,'LS 2-term Period 3rd peak (d)':spinner.p_ls2c,
        'LS 2-term Period 1st amplitude':spinner.a_ls2a,'LS 2-term Period 2nd amplitude':spinner.a_ls2b,'LS 2-term Period 3rd amplitude':spinner.a_ls2c,
        'LS 2-term Period 1st RMS':spinner.rms_ls1a,'LS 2-term Period 1st MAD':spinner.mad_ls1a,
        'LS 2-term Period 2nd RMS':spinner.rms_ls1b,'LS 2-term Period 2nd MAD':spinner.mad_ls1b,
        'LS 2-term Period 3rd RMS':spinner.rms_ls1c,'LS 2-term Period 3rd MAD':spinner.mad_ls1c,
        'ACF Period 1st peak (d)':spinner.p_acfa,'ACF Period 2nd peak (d)':spinner.p_acfb,'ACF Period 3rd peak (d)':spinner.p_acfc,
        'ACF Period 1st amplitude':spinner.a_acfa,'ACF Period 2nd amplitude':spinner.a_acfb,'ACF Period 3rd amplitude':spinner.a_acfc,
        'ACF Period 1st RMS':spinner.rms_ls1a,'ACF Period 1st MAD':spinner.mad_ls1a,
        'ACF Period 2nd RMS':spinner.rms_ls1b,'ACF Period 2nd MAD':spinner.mad_ls1b,
        'ACF Period 3rd RMS':spinner.rms_ls1c,'ACF Period 3rd MAD':spinner.mad_ls1c,
        'LS median power':spinner.ps1_med,'LS 2-term median power':spinner.ps2_med,
        'Rvar':spinner.rvar,'CDPP':spinner.cdpp}

    output_df = pd.DataFrame(output_dict, index=[0])

    cwd = os.getcwd()
    os.chdir(filepath)
    output_df.to_csv(filename)
    os.chdir(cwd)
