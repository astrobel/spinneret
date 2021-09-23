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
        self.cdpp = lc.estimate_cdpp(transit_duration=4).value

    def ls_one_term(self, freq, ps):
        self.freq1 = freq
        self.ps1 = ps
        argmax1 = np.argmax(ps)
        self.p_ls1a = 1/freq[argmax1]
        argmax2 = np.argmax(np.delete(ps, argmax1))
        self.p_ls1b = 1/np.delete(freq, argmax1)[argmax2]
        argmax3 = np.argmax(np.delete(ps, [argmax1, argmax2]))
        self.p_ls1c = 1/np.delete(freq, [argmax1, argmax2])[argmax3]

        self.time_ls1a_fold, self.flux_ls1a_fold, self.orig_time_ls1a_fold, self.model_ls1a = model(self.time, self.flux, self.p_ls1a)
        self.time_ls1b_fold, self.flux_ls1b_fold, self.orig_time_ls1b_fold, self.model_ls1b = model(self.time, self.flux, self.p_ls1b)
        self.time_ls1c_fold, self.flux_ls1c_fold, self.orig_time_ls1c_fold, self.model_ls1c = model(self.time, self.flux, self.p_ls1c)
        self.rms_ls1a = rms(self.model_ls1a, self.flux_ls1a_fold)
        self.rms_ls1b = rms(self.model_ls1b, self.flux_ls1b_fold)
        self.rms_ls1c = rms(self.model_ls1c, self.flux_ls1c_fold)
        self.mad_ls1a = mad(self.model_ls1a, self.flux_ls1a_fold)
        self.mad_ls1b = mad(self.model_ls1b, self.flux_ls1b_fold)
        self.mad_ls1c = mad(self.model_ls1c, self.flux_ls1c_fold)

    def ls_two_term(self, freq, ps):
        self.freq2 = freq
        self.ps2 = ps
        self.p_ls2a = 1/freq[argmax1]
        argmax2 = np.argmax(np.delete(ps2, argmax1))
        self.p_ls2b = 1/np.delete(freq2, argmax1)[argmax2]
        argmax3 = np.argmax(np.delete(ps2, [argmax1, argmax2]))
        self.p_ls2c = 1/np.delete(freq2, [argmax1, argmax2])[argmax3]

        self.time_ls2a_fold, self.flux_ls2a_fold, self.orig_time_ls2a_fold, self.model_ls2a = model(self.time, self.flux, self.p_ls2a)
        self.time_ls2b_fold, self.flux_ls2b_fold, self.orig_time_ls2b_fold, self.model_ls2b = model(self.time, self.flux, self.p_ls2b)
        self.time_ls2c_fold, self.flux_ls2c_fold, self.orig_time_ls2c_fold, self.model_ls2c = model(self.time, self.flux, self.p_ls2c)
        self.rms_ls2a = rms(self.model_ls2a, self.flux_ls2a_fold)
        self.rms_ls2b = rms(self.model_ls2b, self.flux_ls2b_fold)
        self.rms_ls2c = rms(self.model_ls2c, self.flux_ls2c_fold)
        self.mad_ls2a = mad(self.model_ls2a, self.flux_ls2a_fold)
        self.mad_ls2b = mad(self.model_ls2b, self.flux_ls2b_fold)
        self.mad_ls2c = mad(self.model_ls2c, self.flux_ls2c_fold)

    def acf(self, lags, acff):
        self.lags = lags
        self.acf = acff
        self.p_acfa, self.p_acfb, self.p_acfc = get_acf_period(lags, acff)
        # x_peaks, y_peaks = find_all_peaks(lags, acff)
        # period = x_peaks[0]
        # if len(x_peaks) > 2:
        #     good_peaks, errs = find_peaks_at_integers(period, x_peaks)
        #     period, self.p_err = fit_line_to_good_peaks(good_peaks, errs)
        # else:
        #     self.p_err = None
        # self.p_acf = period

        self.time_acfa_fold, self.flux_acfa_fold, self.orig_time_acfa_fold, self.model_acfa = model(self.time, self.flux, self.p_acfa)
        self.time_acfb_fold, self.flux_acfb_fold, self.orig_time_acfb_fold, self.model_acfb = model(self.time, self.flux, self.p_acfb)
        self.time_acfc_fold, self.flux_acfc_fold, self.orig_time_acfc_fold, self.model_acfc = model(self.time, self.flux, self.p_acfc)
        self.rms_acfa = rms(self.model_acfa, self.flux_acfa_fold)
        self.rms_acfb = rms(self.model_acfb, self.flux_acfb_fold)
        self.rms_acfc = rms(self.model_acfc, self.flux_acfc_fold)
        self.mad_acfa = mad(self.model_acfa, self.flux_acfa_fold)
        self.mad_acfb = mad(self.model_acfb, self.flux_acfb_fold)
        self.mad_acfc = mad(self.model_acfc, self.flux_acfc_fold)

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

        xmax = max(self.p_ls1, self.p_ls2, self.p_acf)

        ax['A'].scatter(self.time, self.flux, c=self.time, s=3, cmap=lccmap)# '.', c='#4d0e02', ms='3')
        ax['A'].set(xlabel='time (d)', ylabel='normalized flux', title=heading, xlim=(min(self.time), max(self.time)))

        # 1-term LS
        # ax['B'].axvspan(self.p_ls1-line_ls1(self.p_ls1), self.p_ls1+line_ls1(self.p_ls1), color='#86fa20')
        ax['B'].axvline(self.p_ls1, c='#86fa20', ls='-', lw=10)
        ax['B'].plot(1/self.freq1, self.ps1, c='#4d0e02')
        # ax['B'].axvline(self.p_ls1, c='#33ffbe', ls='--')
        ax['B'].set(xlabel='period (d)', ylabel='power', xlim=(0,xmax+10)) # xscale='log', xlim=(min(1/self.freq1), max(1/self.freq1))

        ax['C'].scatter(self.time_ls1fold, self.flux_ls1fold, c=self.orig_time_ls1fold, s=3, cmap=lccmap)#, '.', c='#4ef5c0', ms='3')
        ax['C'].plot(self.time_ls1fold, self.model_ls1, c='#4d0e02', alpha=0.8, lw=7)
        ax['C'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'LS period: {self.p_ls1:.3f}d', xlim=(min(self.time_ls1fold), max(self.time_ls1fold)))

        # 2-term LS
        # ax['D'].axvspan(self.p_ls2-line_ls2(self.p_ls2), self.p_ls2+line_ls2(self.p_ls2), color='#86fa20')
        ax['D'].axvline(self.p_ls2, c='#86fa20', ls='-', lw=10) # this line will eventually be uncertainty
        ax['D'].plot(1/self.freq2, self.ps2, c='#4d0e02')
        # ax['D'].axvline(self.p_ls2, c='#33ffbe', ls='--')
        ax['D'].set(xlabel='period (d)', ylabel='power', xlim=(0,xmax+10))

        ax['E'].scatter(self.time_ls2fold, self.flux_ls2fold, c=self.orig_time_ls2fold, s=3, cmap=lccmap)#, '.', c='#4ef5c0', ms='3')
        ax['E'].plot(self.time_ls2fold, self.model_ls2, c='#4d0e02', alpha=0.8, lw=7)
        ax['E'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'LS period: {self.p_ls2:.3f}d', xlim=(min(self.time_ls2fold), max(self.time_ls2fold)))

        # ACF
        # ax['F'].axvspan(self.p_acf-line_acf(self.p_acf), self.p_acf+line_acf(self.p_acf), color='#86fa20')
        ax['F'].axvline(self.p_acf, c='#86fa20', ls='-', lw=10) # this line will eventually be uncertainty
        ax['F'].plot(self.lags, self.acf, c='#4d0e02')
        # ax['F'].axvline(self.p_acf, c='#33ffbe', ls='--')
        ax['F'].set(xlim=(0,xmax+10), xlabel='lags', ylabel='ACF')

        ax['G'].scatter(self.time_acffold, self.flux_acffold, c=self.orig_time_affold, s=3, cmap=lccmap)#, '.', c='#ff549b', ms='3')
        ax['G'].plot(self.time_acffold, self.model_acf, c='#4d0e02', alpha=0.8, lw=7)
        ax['G'].set(xlabel='phased time (d)', ylabel='normalized flux', title=f'ACF period: {self.p_acf:.3f}d', xlim=(min(self.time_acffold), max(self.time_acffold)))

        fig.set_size_inches(9,12)

        return fig


def model(t, f, period, terms=1):

    fold = lk.LightCurve(time=t * u.d, flux=f * u.d).fold(period)
    model = ts.LombScargle(fold.time.value, fold.flux.value, nterms=terms).model(fold.time.value, 1/period)

    return fold.time.value, fold.flux.value, fold.time_original.value, model


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


def tessify(lc, sector=14, start_modifier=0):
    """
    Takes a single quarter of Kepler time series data and trims it to match 
    the length and shape of a TESS time series. 

    Args:
        lc (lightkurve LightCurve object): A single quarter of a Kepler
            time series.
        sector (Optional[int]): TESS sector to mimic
        start_modifier (Optional[int]): First cadence to include in the
            TESSified time series.

    Returns:
        lc_new (lightkurve LightCurve object): Single sector TESS-like time series.
    """
    
    # get tess orbit timing
    tess_orbits = pd.read_csv('https://tess.mit.edu/wp-content/uploads/orbit_times_20201013_1338.csv', skiprows=5)
    sectors = tess_orbits['Sector']
    starts = tess_orbits['Start TJD']
    ends = tess_orbits['End TJD']
    
    start1 = starts[sectors==sector].iloc[0]
    end1 = ends[sectors==sector].iloc[0]
    start2 = starts[sectors==sector].iloc[1]
    end2 = ends[sectors==sector].iloc[1]
    span1 = end1-start1
    gap = start2-end1
    span2 = end2-start2
    
    keep = np.zeros(lc.time.value.shape, dtype=bool)

    # get cadence numbers for stop and start points
    try:
        newstart1 = 0 + start_modifier
        newend1 = np.where(np.isclose(lc.time.value,lc.time.value[0] + span1))[0][0] + start_modifier
        newstart2 = np.where(np.isclose(lc.time.value,lc.time.value[newend1] + gap))[0][0] + start_modifier
        newend2 = np.where(np.isclose(lc.time.value,lc.time.value[newstart2] + span2))[0][0] + start_modifier
    except IndexError:
        raise IndexError('Data selected is out of range. Try using a lower start_modifier')
    
    keep[newstart1:newend1+1] = True
    keep[newstart2:newend2+1] = True
    
    lc_new = lk.LightCurve()
    lc_new.time = lc.time[keep==True]
    lc_new.flux = lc.flux[keep==True]
    lc_new.flux_err = lc.flux_err[keep==True]
    lc_new.quality = lc.quality[keep==True]
    
    return lc_new


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
    x_peaks = x[peaks]
    y_peaks = y[peaks]

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

    return xpeaks[0], xpeaks[1], xpeaks[2] # this is the acf period


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


def figsaver(fig, filepath, filename):

    cwd = os.getcwd()
    os.chdir(filepath)
    fig.savefig(filename)
    os.chdir(cwd)
    plt.close(fig)


def filemaker(spinner, kic, p_r, filepath='.', filename=None):
    """
    Args:
        spinner: Spinner object
    """

    if filename == None:
        filename = f'{kic}'

    output_dict = {'KIC':kic,'Santos Period (d)':p_r,
        'LS Period 1st peak (d)':spinner.p_ls1a,'LS Period 2nd peak(d)':spinner.p_ls1b,'LS Period 3rd peak (d)':spinner.p_ls1c,
        'LS Period 1st RMS':spinner.rms_ls1a,'LS Period 1st MAD':spinner.mad_ls1a,
        'LS Period 2nd RMS':spinner.rms_ls1b,'LS Period 2nd MAD':spinner.mad_ls1b,
        'LS Period 3rd RMS':spinner.rms_ls1c,'LS Period 3rd MAD':spinner.mad_ls1c,
        'LS 2-term Period 1st peak (d)':spinner.p_ls2a,'LS 2-term Period 2nd peak (d)':spinner.p_ls2b,'LS 2-term Period 3rd peak (d)':spinner.p_ls2c,
        'LS 2-term Period 1st RMS':spinner.rms_ls1a,'LS 2-term Period 1st MAD':spinner.mad_ls1a,
        'LS 2-term Period 2nd RMS':spinner.rms_ls1b,'LS 2-term Period 2nd MAD':spinner.mad_ls1b,
        'LS 2-term Period 3rd RMS':spinner.rms_ls1c,'LS 2-term Period 3rd MAD':spinner.mad_ls1c,
        'ACF Period 1st peak (d)':spinner.p_acfa,'ACF Period 2nd peak (d)':spinner.p_acfb,'ACF Period 3rd peak (d)':spinner.p_acfc,
        'ACF Period 1st RMS':spinner.rms_ls1a,'ACF Period 1st MAD':spinner.mad_ls1a,
        'ACF Period 2nd RMS':spinner.rms_ls1b,'ACF Period 2nd MAD':spinner.mad_ls1b,
        'ACF Period 3rd RMS':spinner.rms_ls1c,'ACF Period 3rd MAD':spinner.mad_ls1c,
        'Rvar':spinner.rvar,'CDPP':spinner.cdpp}

    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(f'{filepath}{filename}')