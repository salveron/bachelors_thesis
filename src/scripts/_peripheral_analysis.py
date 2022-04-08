#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions to work with Cochleagrams."""

from brian2 import *  # Imports everything from numpy and matplotlib too
from brian2hears import *

from _utils import (MIN_PIANO_KEY_FREQ,
                    MAX_PIANO_KEY_FREQ,
                    _decrease_time_resolution)


def compute_cochleagram(sound, n_channels, min_freq=MIN_PIANO_KEY_FREQ, max_freq=MAX_PIANO_KEY_FREQ,
                        min_bw=50*Hz, max_bw=1*kHz, return_cf=False):
    """Compute a cochleagram from the input sound.

    The cochleagram is computed using filters and filterbanks from Brian2Hears library.
    For more information, visit: https://brian2hears.readthedocs.io/en/stable/index.html

    :param Sound sound: Input sound
    :param int n_channels: Number of frequency channels in the output cochleagram
    :param float min_freq: Lowest frequency value (for piano keys, the lowest value is 27.5 Hz = A0, #1)
    :param float max_freq: Highest frequency value (for piano keys, the highest value is 4.186 kHz = C8, #88)
    :param float min_bw: Lowest frequency channel bandwidth (default = 50 Hz)
    :param float max_bw: Highest frequency channel bandwidth (default = 1 kHz)
    :param bool return_cf: If True, also returns center frequencies of the filters
    :returns: Cochleagram
    :rtype: np.ndarray

    """
    min_freq = float(min_freq) * Hz
    max_freq = float(max_freq) * Hz

    center_freqs = erbspace(min_freq, max_freq, n_channels)
    bandwidth = linspace(min_bw, max_bw, n_channels)
    cutoffs = vstack((center_freqs - bandwidth / 2, center_freqs + bandwidth / 2))

    # Gammatone filterbank
    gammatone = Gammatone(sound, center_freqs)

    # Application of the cubic root function + clipping
    cochlea = FunctionFilterbank(gammatone, lambda x: clip(x, 0, Inf) ** (1.0 / 3.0))

    # Bandpass filtering for every channel
    cochleagram = Butterworth(cochlea, n_channels, order=2, fc=cutoffs, btype="bandpass").process().T

    if return_cf:
        return cochleagram, center_freqs
    else:
        return cochleagram


def plot_cochleagram(cochleagram, samplerate, figsize=(12, 7)):
    """Plot a cochleagram.

    :param np.ndarray cochleagram: Input cochleagram
    :param int samplerate: Samplerate of the input sound
    :param tuple figsize: Size of the matplotlib figure

    """
    rms = _decrease_time_resolution(cochleagram, samplerate)

    figure(figsize=figsize)
    img = imshow(rms, origin='lower', aspect='auto', interpolation="none", vmin=0,
                 extent=[0, cochleagram.shape[1] / samplerate, 0, cochleagram.shape[0]])
    colorbar(img)

    title("Cochleagram")
    xlabel("Time (s)")
    ylabel("Frequency Channels")

    show()
