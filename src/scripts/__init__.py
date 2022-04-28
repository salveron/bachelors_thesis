#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Gathers all functions for the thesis in one place."""

from _sounds import (load_sound,
                     save_sound,
                     add_white_noise,
                     add_other_background,
                     convert_to_binaural)

from _peripheral_analysis import (compute_cochleagram,
                                  plot_cochleagram)

from _feature_extraction import (compute_correlogram,
                                 compute_sacf,
                                 compute_tfcf,
                                 compute_cccf,
                                 find_dominant_lags,
                                 find_fundamental_frequencies,
                                 compute_energy_values,
                                 compute_agreement_values,
                                 plot_correlogram)

from _segmentation_and_grouping import (form_time_segments,
                                        form_frequency_segments,
                                        compute_ibm,
                                        plot_segmentation,
                                        plot_ibm)

from _resynthesis import resynthesize_sound

from _evaluation import (apply_mask,
                         prepare_clean_data,
                         prepare_noised_data,
                         create_dataset,
                         load_dataset,
                         train_classifier,
                         make_prediction,
                         compute_model_accuracy)

from _utils import (MIN_PIANO_KEY_FREQ,
                    MAX_PIANO_KEY_FREQ,
                    WINDOW_SIZE_MS,
                    WINDOW_OVERLAP_MS,
                    load_arr_from_file,
                    save_arr_to_file,
                    compute_lag_boundaries,
                    apply_windowing,
                    plot_process_results)


def process(file_name, load_ibm_from=None,
            save_noised=False, noised_file_path=None,
            save_ibm=False, ibm_file_path=None,
            save_resynth=False, resynth_file_path=None,
            draw_plot=True, plot_title=None,
            save_plot=False, plot_file_path=None, **kwargs):
    """All-at-once function made for experiments.

    :param str file_name: Name of the input .wav file from the data folder
    :param Optional[str] load_ibm_from: If provided, loads a precomputed IBM from the specified file

    :param bool save_noised: If True, saves the noised sound to the specified folder
    :param Optional[str] noised_file_path: Path to the save file for the noised sound

    :param bool save_ibm: If True, saves the resulting ideal binary mask to the specified folder
    :param Optional[str] ibm_file_path: Path to the save file for the resulting plot

    :param bool save_resynth: If True, saves the resynthesized sound to the specified folder
    :param Optional[str] resynth_file_path: Path to the save file for the resynthesized sound

    :param bool draw_plot: If True, draw a plot showing the results
    :param str plot_title: Title of the plot

    :param bool save_plot: If True, saves the resulting plot to the specified folder
    :param Optional[str] plot_file_path: Path to the save file for the resulting plot

    :param dict kwargs: Dictionary with parameters for the algorithms. Described below in more detail.
    :returns: Loaded input sound, the corresponding cochleagram and loaded/computed IBM
    :rtype: tuple

    If some parameters are missing in the input dictionary, they will be set to default values. Here is the list
    of parameters to experiment with:

        - noise_level (float): the amplitude of the white noise added to the input sound (default = no noise).

        - n_channels (int): number of resulting frequency channels in the cochleagram, or the number of
          gammatone filters in the filterbank (default = 128).

        - min_freq, max_freq: lowest and highest center frequency for the gammatone filterbank
          (default = MIN_PIANO_KEY_FREQ, MAX_PIANO_KEY_FREQ)

        _ w_size_ms, w_overlap_ms: size and overlap for the windowing function in milliseconds
          (default = WINDOW_SIZE_MS, WINDOW_OVERLAP_MS)

        _ n_lags (int): number of lags to compute ACF for (default = window size)

        - n_harmonics (int): number of harmonics for F0 estimation. Too low or high numbers may give incorrect
          results (default = 5).

        - energy_threshold (float): threshold for the input RMS sound energy. T-F units with energy lower than
          this number are considered a background. This helps to filter out silent regions (default = 0.05)

        - agreement_threshold (float): threshold for the "agreement ratios", or how much T-F units should agree
          with the T-F unit corresponding to the estimated F0 for the current time frame (default = 0.7)

    """
    from os.path import join as pjoin
    import time
    import warnings
    time_start = time.time()

    if "n_channels" not in kwargs.keys(): kwargs["n_channels"] = 128
    if "min_freq" not in kwargs.keys(): kwargs["min_freq"] = MIN_PIANO_KEY_FREQ
    if "max_freq" not in kwargs.keys(): kwargs["max_freq"] = MAX_PIANO_KEY_FREQ
    if "w_size_ms" not in kwargs.keys(): kwargs["w_size_ms"] = WINDOW_SIZE_MS
    if "w_overlap_ms" not in kwargs.keys(): kwargs["w_overlap_ms"] = WINDOW_OVERLAP_MS
    if "n_harmonics" not in kwargs.keys(): kwargs["n_harmonics"] = 5
    if "energy_threshold" not in kwargs.keys(): kwargs["energy_threshold"] = 0.05
    if "agreement_threshold" not in kwargs.keys(): kwargs["agreement_threshold"] = 0.7

    # Load sound
    sound = load_sound(file_name)
    base_name, extension = file_name.split(".")

    # Add background: loaded sound, if 'bg_file_name' is present, or white noise otherwise
    if "noise_level" in kwargs.keys():
        if "bg_file_name" in kwargs.keys():
            bg_sound = load_sound(pjoin("..", "data", "background_sounds", kwargs["bg_file_name"]), full_path=True)
            sound = add_other_background(sound, bg_sound, kwargs["noise_level"])
        else:
            sound = add_white_noise(sound, kwargs["noise_level"])

    # Save the noised sound
    if save_noised:
        if noised_file_path is None:
            suffix = (("_WN_"
                       if "bg_file_name" not in kwargs.keys()
                       else "_BG" + kwargs["bg_file_name"][9:11] + "_"
                       ) + (",".join(str(kwargs["noise_level"]).split("."))))
            noised_file_path = pjoin("..", "data", "output", base_name + suffix + "." + extension)

        save_sound(sound, file_path=noised_file_path)

    # Compute cochleagram
    print("Cochleagram... ", end="")
    cochleagram, center_freqs = compute_cochleagram(sound,
                                                    n_channels=kwargs["n_channels"],
                                                    min_freq=kwargs["min_freq"],
                                                    max_freq=kwargs["max_freq"],
                                                    return_cf=True)
    print("done! ", end="")

    # Compute IBM, if not loading an already computed one
    if load_ibm_from is None:

        # Apply windowing (rectangular)
        print("Windowing... ", end="")
        windows = apply_windowing(cochleagram,
                                  samplerate=sound.samplerate,
                                  w_size_ms=kwargs["w_size_ms"],
                                  w_overlap_ms=kwargs["w_overlap_ms"])
        print("done! ", end="")

        # Compute correlogram
        print("Correlogram... ", end="")
        correlogram = compute_correlogram(windows,
                                          n_lags=(kwargs["n_lags"] if "n_lags" in kwargs.keys() else None))
        print("done! ", end="")

        # Summary ACF
        sacf = compute_sacf(correlogram)

        # Estimates for F0 and their corresponding lags
        print("F0 estimates... ", end="")
        fundamental_lags, fundamental_freqs = find_fundamental_frequencies(sacf,
                                                                           samplerate=sound.samplerate,
                                                                           n_harmonics=kwargs["n_harmonics"])
        print("done! ", end="")

        # Ideal binary mask estimate
        print("IBM... ", end="")
        ibm = compute_ibm(windows,
                          fundamental_lags,
                          samplerate=sound.samplerate,
                          energy_threshold=kwargs["energy_threshold"],
                          agreement_threshold=kwargs["agreement_threshold"],
                          correlogram=correlogram)
        print("done! ", end="")

        # Save the IBM to a file
        if save_ibm:
            if ibm_file_path is None:
                ibm_file_path = pjoin("..", "data", "masks", base_name + ".npy")
            save_arr_to_file(ibm, file_path=ibm_file_path)

    # Use the precomputed IBM, if provided
    else:
        if save_ibm:
            warnings.warn("The IBM was loaded from an external file."
                          "The save_ibm and ibm_file_path arguments are ignored.")

        print(f"Loading IBM from \"{load_ibm_from}\"... ", end="")
        ibm = load_arr_from_file(load_ibm_from, full_path=True)
        print("done! ", end="")

    # Mask the cochleagram
    if save_resynth or draw_plot:
        print("Masking... ", end="")
        masked_cochleagram = apply_mask(cochleagram,
                                        ibm,
                                        samplerate=sound.samplerate,
                                        w_size_ms=kwargs["w_size_ms"],
                                        w_overlap_ms=kwargs["w_overlap_ms"])
        print("done! ", end="")

    # Resynthesize the sound
    if save_resynth:
        print("Resynthesis... ", end="")
        resynth = resynthesize_sound(masked_cochleagram, samplerate=sound.samplerate)
        print("done! ", end="")

        if resynth_file_path is None:
            suffix = ""
            if "noise_level" in kwargs.keys():
                suffix = (("_WN_"
                           if "bg_file_name" not in kwargs.keys()
                           else "_BG" + kwargs["bg_file_name"][9:11] + "_"
                           ) + (",".join(str(kwargs["noise_level"]).split("."))))
            resynth_file_path = pjoin("..", "data", "output", base_name + suffix + "_resynth." + extension)

        save_sound(resynth, file_path=resynth_file_path)

    # Plot cochleagram, IBM and masked cochleagram
    if draw_plot:
        if plot_file_path is None:
            plot_file_path = pjoin("..", "data", "output", base_name + "_plot.jpg")

        plot_process_results(cochleagram,
                             ibm,
                             masked_cochleagram,
                             samplerate=sound.samplerate,
                             save_figure=save_plot,
                             save_file_path=plot_file_path,
                             figtitle=plot_title)

    # Print execution time
    print(f"\nExecution time: {(time.time() - time_start)} s")

    return sound, cochleagram, ibm


def process_all(file_names, **kwargs):
    """Process all files.

    :param list[str] file_names: Names of the input .wav files from the data folder
    :param dict kwargs: Keyword arguments for the process() function
    :returns: Loaded input sounds, the corresponding cochleagrams and loaded/computed IBMs
    :rtype: tuple

    For a description of supported keyword arguments, refer to the process() function above.

    """
    import time

    sounds, cochleagrams, ibms = [], [], []

    time_start = time.time()

    for file_name in file_names:
        sound, cochleagram, ibm = process(file_name, **kwargs)

        sounds.append(sound)
        cochleagrams.append(cochleagram)
        ibms.append(ibm)

    # Print execution time
    print(f"All files processed. Overall execution time: {(time.time() - time_start)} s")

    return sounds, cochleagrams, ibms
