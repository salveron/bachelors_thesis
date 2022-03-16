#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for Feature Extraction."""

import numpy as np

from _utils import compute_lag_boundaries


def find_dominant_harmonics(sacf, samplerate, n_harmonics=5, return_boundaries=False):
    """Find dominant harmonics in the input summary autocorrelation.

    :param np.ndarray sacf: Input SACF array
    :param int samplerate: Samplerate of the input sound
    :param int n_harmonics: Number of first harmonics to find
    :param bool return_boundaries: If True, also returns min and max values for lags
    :returns: Lag values for the dominant harmonics for each time frame and lag boundaries
    :rtype: tuple

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
    harmonics = find_dominant_harmonics(sacf, samplerate, n_harmonics)

    lags = harmonics[:, 0]
    frequencies = samplerate / lags

    return lags, frequencies
