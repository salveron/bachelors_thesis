#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions to work with Correlograms."""

from statsmodels.tsa.stattools import acf

from brian2 import *  # Imports everything from numpy and matplotlib too
from brian2hears import *

from _feature_extraction import find_dominant_harmonics


def compute_acf(windows, n_lags=None):
    """Compute autocorrelation coefficients (correlogram) from the cochleagram split into windows.

    The correlogram is computed using the **autocorrelation function** and its implementation from the
    `statsmodels.tsa.stattools` library. For more information, visit:
    https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html

    :param np.ndarray windows: Input windows
    :param Optional[int] n_lags: Number of lags for every time frame. If None, then n_lags equals the size of the window
    :returns: Autocorrelation for every time frame and frequency channel (correlogram)
    :rtype: np.ndarray

    """
    if n_lags is None:
        n_lags = windows.shape[2]

    return np.apply_along_axis(lambda vec: acf(vec, nlags=n_lags, fft=False), 2, windows)


def compute_ccc(correlogram):
    """Compute cross-channel correlation coefficients (Pearson's) for the given correlogram.

    Cross-channel correlation is computed for every pair of adjacent frequency channels in every time frame.

    :param correlogram: Input correlogram
    :returns: Cross-channel correlation
    :rtype: np.ndarray

    """
    ccc = np.zeros((correlogram.shape[0], correlogram.shape[1]))

    for _t, _f in np.ndindex(ccc.shape):
        if _f + 1 < correlogram.shape[1]:
            ccc[_t, _f] = np.corrcoef(correlogram[_t, _f], correlogram[_t, _f + 1])[0, 1]

    return ccc


def compute_sacf(correlogram):
    """Compute summary autocorrelation function across all frequency channels.

    SACF is used to determine dominant frequencies in a correlogram.

    :param correlogram: Input correlogram
    :returns: Summary autocorrelation
    :rtype: np.ndarray

    """
    return correlogram.sum(axis=1)


def plot_correlogram(correlogram, window_num, samplerate,
                     show_f0=False, show_harmonics=False, n_harmonics=5, figsize=(12, 10)):
    """Plot correlogram for the input time frame.

    Plots correlogram (autocorrelations) along with corresponding cross-channel correlation and summary
    autocorrelation for the given time frame.

    :param np.ndarray correlogram: Input correlogram
    :param int window_num: Number of a time frame to display
    :param int samplerate: Samplerate of the input sound
    :param bool show_f0: If True, displays the estimated fundamental frequency on the SACF plot
    :param bool show_harmonics: If True, displays `n_harmonics` estimated harmonics for f0
    :param Optional[int] n_harmonics: Number of harmonics to display, if `show_harmonics` is True
    :param tuple figsize: Size of the matplotlib figure

    """
    ccc = compute_ccc(correlogram)
    sacf = compute_sacf(correlogram)

    fig, ((ax1, ax2), (ax3, ax4)) = subplots(ncols=2, nrows=2, figsize=figsize, sharey="row", sharex="col",
                                             gridspec_kw={"width_ratios": [8, 1], "height_ratios": [8, 1]})
    ax4.set_visible(False)
    subplots_adjust(wspace=0.03, hspace=0.03)

    fig.suptitle(f"Correlogram for Time Frame #{window_num}:")

    ax1.imshow(correlogram[window_num], origin='lower', aspect='auto', vmin=0, interpolation="none", cmap='Greys')
    ax1.set_ylabel("Frequency Channels")

    ax2.plot(ccc[window_num], np.arange(correlogram.shape[1]))
    ax2.set_xlabel("CCC")
    ax2.tick_params(axis="x", reset=True)

    ax3.plot(sacf[window_num])
    ax3.set_xlabel("Lags")
    ax3.set_ylabel("SACF")

    if show_harmonics or show_f0:
        harmonics = find_dominant_harmonics(sacf, samplerate, n_harmonics)[window_num]

        if show_harmonics:
            for harmonic in harmonics:
                ax3.axvline(harmonic, color="green", linewidth=0.5)

        if show_f0:
            lag = harmonics[0]
            ax3.scatter(lag, sacf[window_num, lag], color="red")

    show()
