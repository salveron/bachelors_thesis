#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for Segmentation and Grouping."""

import numpy as np

from brian2 import *

from _feature_extraction import (compute_correlogram,
                                 compute_twcf,
                                 compute_cccf,
                                 compute_energy_values,
                                 compute_agreement_ratios)


def form_time_segments(correlogram, twcf_threshold):
    """Form horizontal (across time axis) segments using the correlation between adjacent time windows.

    The segments are formed by segregating together T-F units that have the correlation between adjacent time windows
    higher than the given threshold. When the TWCF between the neighboring units is low, new segment is started.

    :param np.ndarray correlogram: Input correlogram
    :param float twcf_threshold: Threshold for correlation between adjacent time windows
    :returns: Binary matrix representing the formed segments
    :rtype: np.ndarray

    """
    twcf = compute_twcf(correlogram)
    segment_assignments = np.empty((correlogram.shape[0], correlogram.shape[1]))

    for _f in range(segment_assignments.shape[1]):

        # Segment assignment that will be switched when the TWCF is lower than the threshold
        current_segment = True

        for _t in range(segment_assignments.shape[0]):

            # Switch the assignment value when adjacent units don't correlate well enough
            if twcf[_t, _f] < twcf_threshold:
                current_segment = not current_segment

            segment_assignments[_t, _f] = current_segment

    return segment_assignments


def form_frequency_segments(correlogram, cccf_threshold):
    """Form vertical (across frequency axis) segments using cross-channel correlation.

    The segments are formed by segregating together T-F units that have cross-channel correlation higher than
    the given threshold. When the CCCF between the neighboring units is low, new segment is started.

    :param np.ndarray correlogram: Input correlogram
    :param float cccf_threshold: Threshold for cross-channel correlation
    :returns: Binary matrix representing the formed segments
    :rtype: np.ndarray

    """
    cccf = compute_cccf(correlogram)
    segment_assignments = np.empty((correlogram.shape[0], correlogram.shape[1]))

    for _t in range(segment_assignments.shape[0]):

        # Segment assignment that will be switched when the CCCF is lower than the threshold
        current_segment = True

        for _f in range(segment_assignments.shape[1]):

            # Switch the assignment value when adjacent units don't correlate well enough
            if cccf[_t, _f] < cccf_threshold:
                current_segment = not current_segment

            segment_assignments[_t, _f] = current_segment

    return segment_assignments


def compute_ibm(windows, fundamental_lags, samplerate, energy_threshold, agreement_threshold,
                correlogram=None, return_components=False):
    """Compute estimate for ideal binary mask for the input correlogram.

    :param np.ndarray windows: Input windows (cochleagram)
    :param np.ndarray fundamental_lags: Lag values for estimated fundamental frequencies for each time frame
    :param int samplerate: Input sound samplerate
    :param float energy_threshold: Threshold for T-F unit sound energy
    :param float agreement_threshold: Threshold for agreement ratios
    :param Optional[np.ndarray] correlogram: Input correlogram
    :param bool return_components: If True, returns energy values and agreement ratios too
    :returns: Ideal binary mask estimate
    :rtype: np.ndarray

    """
    if correlogram is None:
        correlogram = compute_correlogram(windows)

    energy_values = compute_energy_values(windows)
    agreement_ratios = compute_agreement_ratios(correlogram, fundamental_lags, samplerate)

    # Ideal binary mask is formed from T-F units with high RMS sound energy and high agreement ratios
    # with the estimated fundamental frequencies
    ibm = np.logical_and((energy_values > energy_threshold),
                         (agreement_ratios > agreement_threshold))

    # TODO: try to apply horizontal and vertical segmentation?
    # time_segmentation = form_time_segments(correlogram, twcf_threshold)
    # frequency_segmentation = form_frequency_segments(correlogram, cccf_threshold)

    if return_components:
        return ibm, energy_values, agreement_ratios
    else:
        return ibm


def plot_segmentation(correlogram, twcf_threshold, cccf_threshold, figsize=(14, 5)):
    """Plot segmentation.

    :param np.ndarray correlogram: Input correlogram
    :param float twcf_threshold: Threshold for correlation between adjacent time windows
    :param float cccf_threshold: Threshold for cross-channel correlation
    :param tuple figsize: Size of the matplotlib figure

    """
    time_segmentation = form_time_segments(correlogram, twcf_threshold)
    frequency_segmentation = form_frequency_segments(correlogram, cccf_threshold)

    fig, (ax1, ax2) = subplots(ncols=2, figsize=figsize, sharey="row")

    ax1.imshow(time_segmentation.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    ax1.set_title("Horizontal segmentation (time axis)")
    ax1.set_xlabel("Time frames")
    ax1.set_ylabel("Frequency channels")

    ax2.imshow(frequency_segmentation.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    ax2.set_title("Vertical segmentation (frequency axis)")
    ax2.set_xlabel("Time frames")
    ax2.set_ylabel("Frequency channels")

    show()


def plot_ibm(ibm, energy_values, agreement_ratios, figsize=(12, 12)):
    """Plot an ideal binary mask along with its components.

    :param np.ndarray ibm: Input IBM
    :param np.ndarray energy_values: Energy values for the input IBM
    :param np.ndarray agreement_ratios: Agreement ratios for the input IBM
    :param tuple figsize: Size of the matplotlib figure

    """
    figure(figsize=figsize)
    grid = GridSpec(2, 2, wspace=0.25, hspace=0.25, height_ratios=[1, 2])

    ax1 = subplot(grid[0, 0])
    ax1.imshow(energy_values.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    ax1.set_title("Energy values")
    ax1.set_xlabel("Time frames")
    ax1.set_ylabel("Frequency channels")

    ax2 = subplot(grid[0, 1])
    ax2.imshow(agreement_ratios.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    ax2.set_title("Agreement ratios")
    ax2.set_xlabel("Time frames")
    ax2.set_ylabel("Frequency channels")

    ax3 = subplot(grid[1, :])
    ax3.imshow(ibm.T, origin='lower', aspect='auto', vmin=0, interpolation="none", cmap="Greys")
    ax3.set_title("IBM estimate")
    ax3.set_xlabel("Time frames")
    ax3.set_ylabel("Frequency channels")

    show()
