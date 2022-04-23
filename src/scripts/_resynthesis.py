#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for the Resynthesis stage."""

from brian2hears import Sound


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
