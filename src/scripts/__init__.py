#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Gathers all functions for the thesis in one place."""

from _sounds import (load_sound,
                     save_sound,
                     add_white_noise,
                     convert_to_binaural)

from _cochleagram import (compute_cochleagram,
                          plot_cochleagram)

from _correlogram import (compute_acf,
                          compute_ccc,
                          compute_sacf,
                          plot_correlogram)

from _feature_extraction import (find_dominant_harmonics,
                                 find_fundamental_frequencies)

from _segmentation_and_grouping import (compute_agreement_ratios,
                                        compute_ibm,
                                        plot_ibm)

from _resynthesis import (apply_mask,
                          resynthesize_sound)

from _utils import (MIN_PIANO_KEY_FREQ,
                    MAX_PIANO_KEY_FREQ,
                    LO_FREQ_CHANNEL_BANDWIDTH,
                    HI_FREQ_CHANNEL_BANDWIDTH,
                    WINDOW_SIZE_MS,
                    WINDOW_OVERLAP_MS,
                    compute_lag_boundaries,
                    apply_windowing)
