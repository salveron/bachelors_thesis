#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for Segmentation and Grouping."""

import numpy as np

from brian2 import *

from _correlogram import compute_ccc
from _utils import compute_lag_boundaries


def compute_agreement_ratios(correlogram, fundamental_lags, samplerate):
    """Compute ratios of how much every T-F unit corresponds with the estimated F0 for the current time frame.

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
    for _t, _f in np.ndindex(agreement_ratios.shape):
        agreement_ratios[_t, _f] = correlogram[_t, _f, fundamental_lags[_t]] / max_acf[_t, _f]

    return agreement_ratios


def compute_ibm(correlogram, ccc_threshold, agreement_threshold, fundamental_lags, samplerate):
    """Compute estimate for ideal binary mask for the input correlogram.

    :param np.ndarray correlogram: Input correlogram
    :param float ccc_threshold: Threshold for cross-channel correlation
    :param float agreement_threshold: Threshold for agreement ratios
    :param np.ndarray fundamental_lags: Lag values for estimated fundamental frequencies for each time frame
    :param int samplerate: Input sound samplerate
    :returns: Ideal binary mask estimate
    :rtype: np.ndarray

    """
    ccc = compute_ccc(correlogram)
    agreement_ratios = compute_agreement_ratios(correlogram, fundamental_lags, samplerate)

    # Ideal binary mask is formed from T-F units with high cross-channel correlation and high agreement ratios
    # with the estimated fundamental frequencies
    ibm = np.logical_and((ccc > ccc_threshold), (agreement_ratios > agreement_threshold))

    return ibm


def plot_ibm(ibm, figsize=(10, 7)):
    """Plot an ideal binary mask.

    :param np.ndarray ibm: Input IBM
    :param tuple figsize: Size of the matplotlib figure

    """
    figure(figsize=figsize)
    imshow(ibm.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    suptitle("Ideal Binary Mask (IBM) estimate:")
    xlabel("Time frames")
    ylabel("Frequency channels")
    show()
