#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for the Peripheral Analysis stage."""

from brian2 import *  # Imports everything from numpy and matplotlib too
from brian2hears import *

from _utils import (MIN_PIANO_KEY_FREQ,
                    MAX_PIANO_KEY_FREQ,
                    _decrease_time_resolution)


def compute_cochleagram(sound, n_channels=128, min_freq=MIN_PIANO_KEY_FREQ, max_freq=MAX_PIANO_KEY_FREQ, return_cf=False):
    """Compute a cochleagram from the input sound.

    The cochleagram is computed using filters and filterbanks from Brian2Hears library.
    For more information, visit: https://brian2hears.readthedocs.io/en/stable/index.html

    :param Sound sound: Input sound
    :param int n_channels: Number of frequency channels in the output cochleagram
    :param float min_freq: Lowest frequency value (for piano keys, the lowest value is 27.5 Hz = A0, #1)
    :param float max_freq: Highest frequency value (for piano keys, the highest value is 4.186 kHz = C8, #88)
    :param bool return_cf: If True, also returns center frequencies of the filters
    :returns: Cochleagram
    :rtype: Union[tuple, np.ndarray]

    """
    min_freq = float(min_freq) * Hz
    max_freq = float(max_freq) * Hz

    center_freqs = erbspace(min_freq, max_freq, n_channels)

    # Gammatone filterbank
    gammatone = Gammatone(sound, center_freqs)

    # Clipping (unit step function) + cubic root function to amplify low amplitudes
    clipper = FunctionFilterbank(gammatone, lambda x: clip(x, 0, Inf) ** (1.0 / 3.0))
    cochleagram = clipper.process().T

    if return_cf:
        return cochleagram, center_freqs
    else:
        return cochleagram


def plot_cochleagram(cochleagram, samplerate, figtitle="Cochleagram",
                     figsize=(12, 7), save_figure=False, save_file_path=None):
    """Plot a cochleagram.

    :param np.ndarray cochleagram: Input cochleagram
    :param int samplerate: Samplerate of the input sound
    :param str figtitle: Title of the plot
    :param tuple figsize: Size of the matplotlib figure
    :param bool save_figure: If True, saves the resulting plot to a JPG file
    :param Optional[str] save_file_path: Path to the output file

    """
    rms = _decrease_time_resolution(cochleagram, samplerate)

    fig = figure(figsize=figsize)
    img = imshow(rms, origin='lower', aspect='auto', vmin=0,
                 extent=[0, cochleagram.shape[1] / samplerate, 0, cochleagram.shape[0]])
    colorbar(img)

    title(figtitle, fontsize=14)
    xlabel("Time (s)", fontsize=14)
    ylabel("Frequency Channels", fontsize=14)

    if save_figure:
        if save_file_path is None:
            save_file_path = "cochleagram.jpg"
        fig.savefig(save_file_path, bbox_inches='tight', dpi=384)

    tight_layout()
    show()
