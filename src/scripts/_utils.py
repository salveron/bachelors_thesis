#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains constants and utility functions."""

import skimage

from brian2 import Hz, kHz, msecond

MIN_PIANO_KEY_FREQ = 27.5 * Hz
MAX_PIANO_KEY_FREQ = 4.186 * kHz

LO_FREQ_CHANNEL_BANDWIDTH = 50 * Hz
HI_FREQ_CHANNEL_BANDWIDTH = 1 * kHz

WINDOW_SIZE_MS = 20 * msecond
WINDOW_OVERLAP_MS = 10 * msecond


def compute_lag_boundaries(samplerate, n_lags=None, min_freq=MIN_PIANO_KEY_FREQ, max_freq=MAX_PIANO_KEY_FREQ):
    """Compute min and max values for lags for the given samplerate.

    :param int samplerate: Input samplerate
    :param Optional[int] n_lags: If provided, max value will be clipped if out-of-range
    :param float min_freq: Lowest frequency value (for piano keys, the lowest value is 27.5 Hz = A0, #1)
    :param float max_freq: Highest frequency value (for piano keys, the highest value is 4.186 kHz = C8, #88)
    :returns: Min and max values of lags
    :rtype: tuple

    """
    min_freq = float(min_freq) * Hz
    max_freq = float(max_freq) * Hz

    min_lag = int(samplerate / max_freq)
    if n_lags is None:
        max_lag = int(samplerate / min_freq)
    else:
        max_lag = min(int(samplerate / min_freq), n_lags)

    return min_lag, max_lag


def apply_windowing(cochleagram, samplerate, w_size_ms=WINDOW_SIZE_MS, w_overlap_ms=WINDOW_OVERLAP_MS):
    """Split a cochleagram into windows.

    :param np.ndarray cochleagram: Cochleagram to split
    :param int samplerate: Samplerate of the input sound
    :param int w_size_ms: Size of the window in milliseconds
    :param int w_overlap_ms: Overlap for two adjacent windows in milliseconds
    :returns: Cochleagram split into windows
    :rtype: np.ndarray

    """
    w_size = int(w_size_ms * samplerate)
    w_overlap = int(w_overlap_ms * samplerate)

    window_shape = (cochleagram.shape[0], w_size)

    return skimage.util.view_as_windows(cochleagram, window_shape, w_overlap).squeeze()
