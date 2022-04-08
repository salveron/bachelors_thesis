#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Gathers all functions for the thesis in one place."""

from brian2 import Hz, kHz

from _sounds import (load_sound,
                     save_sound,
                     add_white_noise,
                     convert_to_binaural)

from _peripheral_analysis import (compute_cochleagram,
                                  plot_cochleagram)

from _feature_extraction import (compute_correlogram,
                                 compute_sacf,
                                 compute_twcf,
                                 compute_cccf,
                                 find_dominant_harmonics,
                                 find_fundamental_frequencies,
                                 compute_energy_values,
                                 compute_agreement_ratios,
                                 plot_correlogram)

from _segmentation_and_grouping import (form_time_segments,
                                        form_frequency_segments,
                                        compute_ibm,
                                        plot_segmentation,
                                        plot_ibm)

from _resynthesis import (apply_mask,
                          resynthesize_sound)

from _utils import (MIN_PIANO_KEY_FREQ,
                    MAX_PIANO_KEY_FREQ,
                    WINDOW_SIZE_MS,
                    WINDOW_OVERLAP_MS,
                    compute_lag_boundaries,
                    apply_windowing,
                    plot_process_results)


def process(file_name, save_noised=False, save_resynth=False, **kwargs):
    """All-at-once function made for experiments.

    :param str file_name: Name of the input .wav file from the data folder
    :param bool save_noised: If True, saves the noised sound to the output folder
    :param bool save_resynth: If True, saves the resynthesized sound to the output folder
    :param dict kwargs: Dictionary with parameters for the algorithms. Described below in more detail.

    If some parameters are missing in the input dictionary, they will be set to default values. Here is the list
    of parameters to experiment with:

        - noise_level (float): the amplitude of the white noise added to the input sound (default = no noise).

        - n_channels (int): number of resulting frequency channels in the cochleagram, or the number of
          gammatone filters in the filterbank (default = 128).

        - min_freq, max_freq: lowest and highest center frequency for the gammatone filterbank
          (default = MIN_PIANO_KEY_FREQ, MAX_PIANO_KEY_FREQ)

        - min_bw, max_bw: bandwidths for the lowest and highest gammatone filters in the filterbank
          (default = 50 Hz, 1 kHz)

        _ w_size_ms, w_overlap_ms: size and overlap for the windowing function in milliseconds
          (default = WINDOW_SIZE_MS, WINDOW_OVERLAP_MS)

        _ n_lags (int): number of lags to compute ACF for (default = window size)

        - n_harmonics (int): number of harmonics for F0 estimation. Too low or high numbers may give incorrect
          results (default = 5).

        - energy_threshold (float): threshold for the input RMS sound energy. T-F units with energy lower than this number
          are considered a background. This helps to filter out silent regions (default = 0.05)

        - agreement_threshold (float): threshold for the "agreement ratios", or how much T-F units should agree with the
          T-F unit corresponding to the estimated F0 for the current time frame (default = 0.7)

    """
    if "n_channels" not in kwargs.keys(): kwargs["n_channels"] = 128
    if "min_freq" not in kwargs.keys(): kwargs["min_freq"] = MIN_PIANO_KEY_FREQ
    if "max_freq" not in kwargs.keys(): kwargs["max_freq"] = MAX_PIANO_KEY_FREQ
    if "min_bw" not in kwargs.keys(): kwargs["min_bw"] = 50 * Hz
    if "max_bw" not in kwargs.keys(): kwargs["max_bw"] = 1 * kHz
    if "w_size_ms" not in kwargs.keys(): kwargs["w_size_ms"] = WINDOW_SIZE_MS
    if "w_overlap_ms" not in kwargs.keys(): kwargs["w_overlap_ms"] = WINDOW_OVERLAP_MS
    if "n_harmonics" not in kwargs.keys(): kwargs["n_harmonics"] = 5
    if "energy_threshold" not in kwargs.keys(): kwargs["energy_threshold"] = 0.05
    if "agreement_threshold" not in kwargs.keys(): kwargs["agreement_threshold"] = 0.7

    # Load sound
    sound = load_sound(file_name)
    base_name, extension = file_name.split(".")

    # Add white noise
    if "noise_level" in kwargs.keys():
        sound = add_white_noise(sound, kwargs["noise_level"])

    # Save the noised sound
    if save_noised:
        noised_file_name = base_name + " Noised." + extension
        save_sound(sound, noised_file_name)

    # Compute cochleagram
    cochleagram, center_freqs = compute_cochleagram(sound,
                                                    n_channels=kwargs["n_channels"],
                                                    min_freq=kwargs["min_freq"],
                                                    max_freq=kwargs["max_freq"],
                                                    min_bw=kwargs["min_bw"],
                                                    max_bw=kwargs["max_bw"],
                                                    return_cf=True)

    # Apply windowing (rectangular)
    windows = apply_windowing(cochleagram,
                              samplerate=sound.samplerate,
                              w_size_ms=kwargs["w_size_ms"],
                              w_overlap_ms=kwargs["w_overlap_ms"])

    # Compute correlogram
    correlogram = compute_correlogram(windows,
                                      n_lags=(kwargs["n_lags"]
                                              if "n_lags" in kwargs.keys()
                                              else None))

    # Summary ACF
    sacf = compute_sacf(correlogram)

    # Estimates for F0 and their corresponding lags
    fundamental_lags, fundamental_freqs = find_fundamental_frequencies(sacf,
                                                                       samplerate=sound.samplerate,
                                                                       n_harmonics=kwargs["n_harmonics"])

    # Ideal binary mask estimate
    ibm = compute_ibm(windows,
                      fundamental_lags,
                      samplerate=sound.samplerate,
                      energy_threshold=kwargs["energy_threshold"],
                      agreement_threshold=kwargs["agreement_threshold"],
                      correlogram=correlogram)

    # Mask the cochleagram
    masked_cochleagram = apply_mask(cochleagram,
                                    ibm,
                                    samplerate=sound.samplerate,
                                    w_size_ms=kwargs["w_size_ms"],
                                    w_overlap_ms=kwargs["w_overlap_ms"])

    # Resynthesize the sound
    if save_resynth:
        resynth = resynthesize_sound(masked_cochleagram, samplerate=sound.samplerate)
        save_sound(resynth, base_name + " Resynth." + extension)

    # Plot cochleagram, IBM and masked cochleagram
    plot_process_results(cochleagram,
                         ibm,
                         masked_cochleagram,
                         samplerate=sound.samplerate)
