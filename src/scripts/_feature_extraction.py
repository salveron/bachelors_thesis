#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for Feature Extraction."""

from brian2 import *  # Imports everything from numpy and matplotlib too
from brian2hears import *
from statsmodels.tsa.stattools import acf

from _utils import compute_lag_boundaries


def compute_correlogram(windows, n_lags=None):
    """Compute autocorrelation coefficients (correlogram) from the cochleagram split into windows.

    The correlogram is computed using the **autocorrelation function** and its implementation from the
    `statsmodels.tsa.stattools` library. For more information, visit:
    https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html

    :param np.ndarray windows: Input windows (cochleagram)
    :param Optional[int] n_lags: Number of lags for every time frame. If None, then n_lags equals the size of the window
    :returns: Autocorrelation for every time frame and frequency channel (correlogram)
    :rtype: np.ndarray

    """
    if n_lags is None:
        n_lags = windows.shape[2]

    return np.apply_along_axis(lambda vec: acf(vec, nlags=n_lags, missing="conservative"), 2, windows)


def compute_sacf(correlogram):
    """Compute summary autocorrelation function across all frequency channels.

    SACF is used to determine dominant frequencies in a correlogram.

    :param correlogram: Input correlogram
    :returns: Summary autocorrelation
    :rtype: np.ndarray

    """
    return correlogram.sum(axis=1)


def compute_twcf(windows):
    """Compute correlation coefficients (Pearson's) between adjacent time windows for the given correlogram.

    The correlation is computed for every pair of T-F units adjacent in time in every frequency channel.

    :param windows: Input windows (cochleagram)
    :returns: Correlation for adjacent time windows
    :rtype: np.ndarray

    """
    twcf = np.zeros((windows.shape[0], windows.shape[1]))

    for _t, _f in np.ndindex(twcf.shape):
        if _t + 1 < windows.shape[0]:
            twcf[_t, _f] = np.corrcoef(windows[_t, _f], windows[_t + 1, _f])[0, 1]

    return twcf


def compute_cccf(windows):
    """Compute cross-channel correlation coefficients (Pearson's) for the given correlogram.

    Cross-channel correlation is computed for every pair of T-F units adjacent in frequency in every time frame.

    :param windows: Input windows (cochleagram)
    :returns: Cross-channel correlation
    :rtype: np.ndarray

    """
    cccf = np.zeros((windows.shape[0], windows.shape[1]))

    for _t, _f in np.ndindex(cccf.shape):
        if _f + 1 < windows.shape[1]:
            cccf[_t, _f] = np.corrcoef(windows[_t, _f], windows[_t, _f + 1])[0, 1]

    return cccf


def find_dominant_lags(sacf, samplerate, n_harmonics=5, return_boundaries=False):
    """Find dominant harmonics in the input summary autocorrelation.

    :param np.ndarray sacf: Input SACF array
    :param int samplerate: Samplerate of the input sound
    :param int n_harmonics: Number of first harmonics to find
    :param bool return_boundaries: If True, also returns min and max values for lags
    :returns: Lag values for the dominant harmonics for each time frame and lag boundaries
    :rtype: Union[np.ndarray, tuple]

    """
    # Lag boundaries for piano keys
    min_lag, max_lag = compute_lag_boundaries(samplerate, n_lags=sacf.shape[1])

    # All possible values for n_harmonics harmonics for input SACF
    harmonics = np.vstack([np.arange(min_lag, max_lag // n_harmonics) * harmonic
                           for harmonic in range(1, n_harmonics + 1)]).T

    # Dominant harmonics are usually where a sum of SACF values is highest
    dominant_harmonics = harmonics[np.argmax(sacf[:, harmonics].sum(axis=2), axis=1)]

    if return_boundaries:
        return dominant_harmonics, min_lag, max_lag
    else:
        return dominant_harmonics


def find_fundamental_frequencies(sacf, samplerate, n_harmonics=5):
    """Find fundamental frequencies estimates in the input summary autocorrelation.

    :param np.ndarray sacf: Input SACF array
    :param int samplerate: Samplerate of the input sound
    :param int n_harmonics: Number of first harmonics to use
    :returns: Lag and frequency values for the estimated fundamental frequency (f0) for every time frame
    :rtype: tuple

    """
    harmonics = find_dominant_lags(sacf, samplerate, n_harmonics)

    lags = harmonics[:, 0]
    frequencies = samplerate / lags

    return lags, frequencies


def compute_energy_values(windows):
    """Compute RMS values for sound energy in T-F units.

    This function helps to filter out the silence.

    :param np.ndarray windows: Input windows (cochleagram)
    :returns: RMS values for sound energy
    :rtype: np.ndarray

    """
    return np.apply_along_axis(lambda vec: vec.mean(), 2, windows)


def compute_agreement_values(correlogram, fundamental_lags, samplerate):
    """Compute estimates of how much every T-F unit corresponds with the estimated F0 for the current time frame.

    The values are taken from the correlogram and normalized by the max value for the given T-F unit.

    :param np.ndarray correlogram: Input correlogram
    :param np.ndarray fundamental_lags: Lag values for estimated fundamental frequencies for each time frame
    :param int samplerate: Input sound samplerate
    :returns: Agreement ratios
    :rtype: np.ndarray

    """
    min_lag, max_lag = compute_lag_boundaries(samplerate, n_lags=correlogram.shape[2])

    # Max values of autocorrelation for each T-F unit
    max_acf = correlogram[:, :, min_lag:max_lag].max(axis=2)

    # Measures of how much each T-F unit is in agreement with the fundamental lag value for the current time frame
    agreement_ratios = np.zeros_like(max_acf)
    for _n, _c in np.ndindex(agreement_ratios.shape):
        agreement_ratios[_n, _c] = correlogram[_n, _c, fundamental_lags[_n]] / max_acf[_n, _c]

    return agreement_ratios


def plot_correlogram(correlogram, window_num, samplerate,
                     cccf=None, sacf=None, show_f0=False, show_harmonics=False, n_harmonics=5, figsize=(12, 10),
                     save_figure=False, save_file_path=None):
    """Plot correlogram for the input time frame.

    Plots correlogram (autocorrelations) along with corresponding cross-channel correlation and summary
    autocorrelation for the given time frame.

    :param np.ndarray correlogram: Input correlogram
    :param int window_num: Number of a time frame to display
    :param int samplerate: Samplerate of the input sound
    :param Optional[np.ndarray] cccf: Cross-channel correlation to display
    :param Optional[np.ndarray] sacf: Summary autocorrelation to display
    :param bool show_f0: If True, displays the estimated fundamental frequency on the SACF plot
    :param bool show_harmonics: If True, displays `n_harmonics` estimated harmonics for f0
    :param Optional[int] n_harmonics: Number of harmonics to display, if `show_harmonics` is True
    :param tuple figsize: Size of the matplotlib figure
    :param bool save_figure: If True, saves the resulting plot to a JPG file
    :param Optional[str] save_file_path: Path to the output file

    """
    if cccf is None:
        cccf = compute_cccf(correlogram)
    if sacf is None:
        sacf = compute_sacf(correlogram)

    fig, ((ax1, ax2), (ax3, ax4)) = subplots(ncols=2, nrows=2, figsize=figsize, sharey="row", sharex="col",
                                             gridspec_kw={"width_ratios": [8, 1], "height_ratios": [8, 1]})
    ax4.set_visible(False)
    subplots_adjust(wspace=0.05, hspace=0.05)

    ax1.imshow(correlogram[window_num], origin='lower', aspect='auto', vmin=0, interpolation="none", cmap='Greys')
    ax1.set_title(f"Correlogram for Time Frame #{window_num}", fontsize=16)
    ax1.set_ylabel("Frequency Channels", fontsize=14)

    ax2.plot(cccf[window_num], np.arange(correlogram.shape[1]))
    ax2.set_xlabel("CCCF", fontsize=14)
    ax2.tick_params(axis="x", reset=True)

    ax3.plot(sacf[window_num])
    ax3.set_xlabel("Lags", fontsize=14)
    ax3.set_ylabel("SACF", fontsize=14)

    if show_harmonics or show_f0:
        harmonics = find_dominant_lags(sacf, samplerate, n_harmonics)[window_num]

        if show_harmonics:
            for harmonic in harmonics:
                ax3.axvline(harmonic, color="green", linewidth=0.5)

        if show_f0:
            lag = harmonics[0]
            ax3.scatter(lag, sacf[window_num, lag], color="red")

    if save_figure:
        if save_file_path is None:
            save_file_path = "correlogram.jpg"
        fig.savefig(save_file_path, bbox_inches='tight', dpi=384)

    show()
