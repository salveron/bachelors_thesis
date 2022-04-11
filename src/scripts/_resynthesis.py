#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for Resynthesis."""

import numpy as np

from brian2hears import Sound

from _utils import (WINDOW_SIZE_MS,
                    WINDOW_OVERLAP_MS,
                    apply_windowing)


def apply_mask(cochleagram, ibm, samplerate, w_size_ms=WINDOW_SIZE_MS, w_overlap_ms=WINDOW_OVERLAP_MS):
    """Mask a cochleagram using the estimated IBM.

    Since IBM was computed for time frames (not separate samples), the cochleagram needs to be
    split into windows first to be rebuilt using new values later.

    :param np.ndarray cochleagram: Input cochleagram
    :param np.ndarray ibm: Estimated IBM
    :param int samplerate: Input sound samplerate
    :param int w_size_ms: Size of the window (number of samples in a window along the time axis)
    :param int w_overlap_ms: Overlap for two adjacent windows
    :returns: Masked cochleagram
    :rtype: np.ndarray

    """
    windows = apply_windowing(cochleagram, samplerate, w_size_ms, w_overlap_ms)

    masked_cochleagram = np.zeros_like(cochleagram)
    for _t in range(ibm.shape[0]):

        _t_lo = int(_t * (w_size_ms - w_overlap_ms) * samplerate)
        _t_mid = int(_t_lo + w_overlap_ms * samplerate)
        _t_hi = int(_t_lo + w_size_ms * samplerate)

        # Overlapped part at the beginning
        masked_cochleagram[:, _t_lo:_t_mid] += np.multiply(windows[_t, :, :(_t_mid - _t_lo)].T, ibm[_t]).T
        if _t != 0:
            masked_cochleagram[:, _t_lo:_t_mid] /= 2

        # The leftovers
        masked_cochleagram[:, _t_mid:_t_hi] = np.multiply(windows[_t, :, (_t_mid - _t_lo):].T, ibm[_t]).T

    return masked_cochleagram


def resynthesize_sound(cochleagram, samplerate):
    """Resynthesize sound using the given cochleagram.

    :param np.ndarray cochleagram: Input cochleagram
    :param int samplerate: Input sound samplerate
    :returns: Resynthesized sound
    :rtype: Sound

    """
    # FUTURE: Invert cochleagram according to Weintraub?
    resynth = Sound(cochleagram.mean(axis=0), samplerate)

    return resynth
