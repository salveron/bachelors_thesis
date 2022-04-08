#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions to work with Sounds."""

import os

import numpy as np
from brian2hears import Sound


def load_sound(file_name, full_path=False, monaural=True):
    """Load a sound from a .wav file.

    For this thesis, sounds are monaural. For binaural input sounds, only left channel is used.

    :param str file_name: Name of the input .wav file from the data folder
    :param bool full_path: If True, file_name is considered to be a valid relative path to the file
    :param bool monaural: If True, only left channel is used as a result
    :returns: Loaded sound
    :rtype: Sound

    """
    if not full_path:
        file_name = os.path.join("..", "data", "wav", file_name)

    if not file_name.endswith(".wav"):
        raise ValueError("Only .wav files are supported.")

    sound = Sound.load(file_name)

    if monaural and sound.nchannels == 2:
        sound = sound.left

    print(f"Loaded sound \"{file_name}\"\n"
          f"Duration: {sound.duration}, samples: {sound.nsamples}, samplerate: {sound.samplerate}")

    return sound


def save_sound(sound, file_name, full_path=False):
    """Save a sound to a .wav file.

    :param Sound sound: Sound to save
    :param str file_name: Name of the input .wav file from the data folder
    :param bool full_path: If True, file_name is considered to be a valid relative path to the file

    """
    if not full_path:
        file_name = os.path.join("..", "data", "output", file_name)

    if not file_name.endswith(".wav"):
        raise ValueError("Only .wav files are supported.")

    sound.save(file_name)


def add_white_noise(sound, noise_level):
    """Add white noise to the input sound.

    :param Sound sound: Input sound
    :param float noise_level: Noise level
    :returns: Sound with noise
    :rtype: Sound

    """
    return sound + noise_level * Sound.whitenoise(sound.duration, sound.samplerate, sound.nchannels)


def convert_to_binaural(sound):
    """Convert monaural sound to binaural by copying the channel.

    :param Sound sound: Input sound
    :returns: Binaural sound
    :rtype: Sound

    """
    return Sound(np.hstack([sound.data, sound.data]), sound.samplerate)


