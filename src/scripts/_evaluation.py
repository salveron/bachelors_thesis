#######################################################
# Bachelor's Thesis. Nikita Mortuzaiev, FIT CVUT, 2022
#######################################################
"""Contains functions for the evaluation of the system."""

from os.path import join as pjoin
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.tree import DecisionTreeClassifier

from _sounds import (load_sound,
                     add_white_noise,
                     add_other_background)
from _peripheral_analysis import compute_cochleagram
from _utils import (WINDOW_SIZE_MS,
                    WINDOW_OVERLAP_MS,
                    apply_windowing,
                    load_arr_from_file,
                    save_arr_to_file,
                    _decrease_time_resolution)


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


def prepare_clean_data(file_names, sounds=None, cochleagrams=None, ibms=None):
    """Prepare a dataset of cochleagrams computed from clean sounds (without noise or other background).

    This function loads sounds (if not provided), computes cochleagrams (if not provided), loads ideal binary masks
    (if not provided), then masks the cochleagrams, flattens them to make single rows of floats, pads with zeros
    to make equal number of dimensions, and stacks them into a single matrix to create the dataset.

    'labels' is the variable for prediction.

    :param list[str] file_names: Names of all available files in the dataset
    :param list[Sound] sounds: Pre-loaded sounds (for optimization)
    :param list[np.ndarray] cochleagrams: Pre-computed cochleagrams (for optimization)
    :param list[np.ndarray] ibms: Pre-loaded IBMs (for optimization)
    :returns: Data matrix (each row is a flattened cochleagram padded with zeros on the right) and the labels
    :rtype: tuple

    """
    if sounds is None:
        sounds = [load_sound(file_name) for file_name in file_names]
    if cochleagrams is None:
        cochleagrams = [compute_cochleagram(sound) for sound in sounds]
    if ibms is None:
        ibms = [load_arr_from_file(file_name.split(".")[0] + ".npy") for file_name in file_names]

    labels = np.array(range(len(file_names)), dtype="int16")
    masked = [_decrease_time_resolution(
        apply_mask(cochleagrams[label], ibms[label], sounds[label].samplerate),
        sounds[label].samplerate
    ) for label in labels]

    data = np.zeros((len(masked),
                     max([cochleagram.shape[0] * cochleagram.shape[1] for cochleagram in masked])),
                    dtype="float32")

    for label, masked in zip(labels, masked):
        flat = masked.flatten()
        data[label, :flat.shape[0]] = flat

    return data, labels


def prepare_noised_data(file_names, n_samples, n_features, sounds=None, ibms=None,
                        use_white_noise=True, noise_level_range=(0.0, 1.0),
                        bg_file_names=None, mask_samples=True, print_sound_stats=False):
    """Prepare a dataset of cochleagrams computed from sounds with added random amounts of white noise.

    This function picks random sounds, adds random amounts of white noise to them, computes corresponding
    cochleagrams, loads ideal binary masks (if not provided), then masks the cochleagrams, flattens them to make
    single rows of floats, pads with zeros to make equal number of dimensions, and stacks them into a single
    matrix to create the dataset.

    'labels' is the variable for prediction.

    :param list[str] file_names: Names of all available files in the dataset
    :param int n_samples: Number of noised cochleagram samples to generate
    :param int n_features: Number of features in the dataset, or the number which needs to be reached by padding
    :param list[Sound] sounds: Pre-loaded sounds (for optimization)
    :param list[np.ndarray] ibms: Pre-loaded IBMs (for optimization)
    :param bool use_white_noise: If True, white noise is added. If False, other random background is chosen
    :param tuple noise_level_range: Low and high limits for background amplitude (chosen uniformly from this interval)
    :param Optional[list] bg_file_names: List of names of files with other background sounds (if not 'use_white_noise')
    :param bool mask_samples: If True, the cochleagrams are masked before being flattened and padded
    :param bool print_sound_stats: If True, `load_sound` function prints the sound stats after loading
    :returns: Data matrix (each row is a flattened cochleagram padded with zeros on the right) and the labels
    :rtype: tuple

    """
    if sounds is None:
        sounds = [load_sound(file_name, print_stats=print_sound_stats) for file_name in file_names]
    if ibms is None:
        ibms = [load_arr_from_file(file_name.split(".")[0] + ".npy") for file_name in file_names]

    data, labels = np.zeros((n_samples, n_features), dtype="float32"), np.empty((n_samples,), dtype="int16")

    for i in range(n_samples):
        label = random.choice(range(len(sounds)))
        sound = sounds[label]

        noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
        if use_white_noise:
            noised_sound = add_white_noise(sound, noise_level=noise_level)
        else:
            if bg_file_names is None:
                raise ValueError("If 'use_white_noise' is False, 'bg_file_names' should be provided.")

            bg_sound = load_sound(pjoin("..", "data", "background_sounds",
                                        random.choice(bg_file_names)),
                                  full_path=True,
                                  print_stats=print_sound_stats)
            noised_sound = add_other_background(sound, bg_sound, noise_level)

        cochleagram = compute_cochleagram(noised_sound)

        if mask_samples:
            cochleagram = apply_mask(cochleagram, ibms[label], sound.samplerate)

        flat = _decrease_time_resolution(cochleagram, sound.samplerate).flatten()
        data[i, :flat.shape[0]] = flat
        labels[i] = label

    return data, labels


def create_dataset(file_names, white_noise_samples=0, other_bg_samples=0, mask_samples=True,
                   sounds=None, cochleagrams=None, ibms=None, bg_file_names=None, noise_level_range=(0.0, 1.0),
                   save_to_file=True, data_file_path=None, labels_file_path=None, print_sound_stats=False):
    """Create a dataset of clean and noised data samples.

    This function creates a unified dataset of clean data (one sample for each input file; masked with
    the pre-computed IBM) and data with backgrounds (the number of samples is specified by `noised_samples`; with random
    noise levels; whether they should be masked determined by `mask_samples`). After the dataset is built, it is saved
    to a file by default.

    'labels' is the variable for prediction.

    :param list[str] file_names: Names of all available files in the dataset
    :param int white_noise_samples: Number of noised samples to append to the dataset
    :param int other_bg_samples: Number of samples with other backgrounds to append to the dataset
    :param bool mask_samples: If True, the cochleagrams for noised data are masked before being flattened and padded
    :param list[Sound] sounds: Pre-loaded sounds (for optimization)
    :param list[np.ndarray] cochleagrams: Pre-computed cochleagrams (for optimization)
    :param list[np.ndarray] ibms: Pre-loaded IBMs (for optimization)
    :param Optional[list] bg_file_names: List of names of files with other background sounds (if 'other_bg_samples' > 0)
    :param tuple noise_level_range: Low and high limits for background amplitude (chosen uniformly from this interval)
    :param bool save_to_file: If True, the generated dataset and the corresponding labels are saved to files
    :param Optional[str] data_file_path: Path to the save file for the data
    :param Optional[str] labels_file_path: Path to the save file for the labels
    :param bool print_sound_stats: If True, `load_sound` function prints the sound stats after loading
    :returns: Data matrix (each row is a flattened cochleagram padded with zeros on the right) and the labels
    :rtype: tuple

    """
    time_start = time.time()

    data, labels = prepare_clean_data(file_names, sounds, cochleagrams, ibms)

    if white_noise_samples > 0:
        noised_data, noised_labels = prepare_noised_data(file_names, white_noise_samples, data.shape[1],
                                                         sounds, ibms, mask_samples=mask_samples,
                                                         use_white_noise=True, noise_level_range=noise_level_range,
                                                         print_sound_stats=print_sound_stats)

        data = np.vstack([data, noised_data])
        labels = np.hstack([labels, noised_labels])

    if other_bg_samples > 0:
        if bg_file_names is None:
            raise ValueError("If 'other_bg_samples' > 0, 'bg_file_names' should be provided.")

        noised_data, noised_labels = prepare_noised_data(file_names, other_bg_samples, data.shape[1],
                                                         sounds, ibms, mask_samples=mask_samples,
                                                         use_white_noise=False, noise_level_range=noise_level_range,
                                                         bg_file_names=bg_file_names,
                                                         print_sound_stats=print_sound_stats)

        data = np.vstack([data, noised_data])
        labels = np.hstack([labels, noised_labels])

    if save_to_file:
        if data_file_path is None:
            data_file_path = pjoin("..", "data", ("" if mask_samples else "un") + f"masked_data.npy")
        save_arr_to_file(data, data_file_path)
        if labels_file_path is None:
            labels_file_path = pjoin("..", "data", ("" if mask_samples else "un") + f"masked_labels.npy")
        save_arr_to_file(labels, labels_file_path)

    time_end = time.time()
    print(f"Dataset created. Execution time: {time_end - time_start} s")

    return data, labels


def load_dataset(data_file_path=None, labels_file_path=None):
    """Load data and labels from files.

    :param Optional[str] data_file_path: Path to the save file for the data
    :param Optional[str] labels_file_path: Path to the save file for the labels
    :returns: Data matrix (each row is a flattened cochleagram padded with zeros on the right) and their labels
    :rtype: tuple

    """
    if data_file_path is None:
        data_file_path = pjoin("..", "data", f"masked_data.npy")
    if labels_file_path is None:
        labels_file_path = pjoin("..", "data", f"masked_labels.npy")

    data = load_arr_from_file(data_file_path, full_path=True)
    labels = load_arr_from_file(labels_file_path, full_path=True)

    print(f"Loaded dataset of shape {data.shape} (dtype={data.dtype}) "
          f"and corresponding labels of shape {labels.shape} (dtype={labels.dtype})")

    return data, labels


def train_classifier(data, labels, validation_size=0.3, random_state=None, **kwargs):
    """Train a simple decision tree for further cochleagram classification.

    The data is firstly split into train and validation sets. The size of the validation set is determined by
    `validation_size`. Then, the best hyperparameters for the model are determined using an iterative approach
    (no cross-validation). The model is then trained with these hyperparameters.

    DecisionTreeClassifier from `sklearn` is used. For further info, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    :param np.ndarray data: Input dataset
    :param np.ndarray labels: Input labels (the variable for prediction)
    :param float validation_size: The percentage of data split from the input dataset for validation
    :param Optional[int] random_state: If True, the splits are always the same
    :param dict kwargs: Arguments for the model (these override the best found hyperparameters)
    :returns: Trained decision tree classifier
    :rtype: DecisionTreeClassifier

    """
    data_train, data_validation, labels_train, labels_validation = train_test_split(data, labels,
                                                                                    test_size=validation_size,
                                                                                    random_state=random_state)

    model_class = DecisionTreeClassifier

    param_grid = {
        "max_depth": list(range(1, 11)),
        "criterion": ["entropy", "gini"]
    }
    param_combinations = ParameterGrid(param_grid)

    time_start_hp = time.time()
    val_acc = np.array([
        accuracy_score(labels_validation,
                       model_class(**(params | kwargs)).fit(data_train, labels_train).predict(data_validation))
        for params in param_combinations])
    best_params = param_combinations[np.argmax(val_acc)]
    time_end_hp = time.time()

    model = model_class(**(best_params | kwargs))

    time_start_tr = time.time()
    model.fit(data_train, labels_train)
    time_end_tr = time.time()

    train_acc = accuracy_score(labels_train, model.predict(data_train))
    validation_acc = accuracy_score(labels_validation, model.predict(data_validation))

    print(f"Trained model: {model_class.__name__}\n"
          f"Best hyperparameters: {best_params}, search time: {time_end_hp - time_start_hp} s\n"
          f"Training time: {time_end_tr - time_start_tr} s\n"
          f"Accuracy scores: train - {train_acc:.2%} ({data_train.shape[0]} samples), "
          f"validation - {validation_acc:.2%} ({data_validation.shape[0]} samples)")

    return model


def make_prediction(model, n_features, file_name=None, sound=None, cochleagram=None, ibm=None, samplerate=None,
                    noise_level=None, bg_file_name=None, use_mask=False, file_names=None):
    """Make a prediction using the trained model.

    One of the following should be provided:

      - name of the input file - 'file_name'
      - pre-loaded sound - 'sound'
      - pre-computed cochleagram and the samplerate of the sound in it - 'cochleagram' and 'samplerate'

    Using this input, the function adds a random amount of white noise to the input sound (if non-zero),
    computes its cochleagram, masks it (if needed), then flattens and pads it and passes it to the trained classifier
    for prediction. The output from the classifier is either interpreted as a name of the sound input file (thus
    'file_names' should be provided), or returns the label as an integer - index of the sound file.

    :param Any model: The trained classifier
    :param int n_features: Number of features (can't determine, because cochleagrams have different dimensions)
    :param Optional[str] file_name: Name of the input sound file. Must be provided, if 'use_mask' is True.
    :param Optional[Sound] sound: Pre-loaded sound (for optimization).
    :param Optional[np.ndarray] cochleagram: Pre-computed cochleagram (for optimization).
    :param Optional[np.ndarray] ibm: Pre-loaded IBM (for optimization).
    :param Optional[int] samplerate: Samplerate of the input sound
    :param float noise_level: Noise level (either for white noise or other background)
    :param Optional[str] bg_file_name: Name of the file with an alternate background sound
    :param bool use_mask: If True, the cochleagram is masked before the prediction
    :param list[str] file_names: If provided, the prediction is returned as a name of the input sound file
    :returns: Class assignment for the cochleagram, corresponding to a scale or an interval progression
    :rtype: Union[str, int]

    """
    if cochleagram is None or samplerate is None:
        if sound is None:
            if file_name is None:
                raise ValueError("Either 'file_name' or 'sound' "
                                 "or ('cochleagram' and 'samplerate') should be provided.")

            sound = load_sound(file_name, print_stats=False)

        if noise_level is not None:
            if bg_file_name is None:
                sound = add_white_noise(sound, noise_level)
            else:
                bg_sound = load_sound(pjoin("..", "data", "background_sounds", bg_file_name),
                                      full_path=True, print_stats=False)
                sound = add_other_background(sound, bg_sound, noise_level)

        if cochleagram is None:
            cochleagram = compute_cochleagram(sound)
        if samplerate is None:
            samplerate = sound.samplerate

    if use_mask:
        if file_name is None:
            raise ValueError("If 'use_mask' is True, 'file_name' should be provided.")
        if ibm is None:
            ibm = load_arr_from_file(file_name.split(".")[0] + ".npy")
        cochleagram = apply_mask(cochleagram, ibm, samplerate)

    flat = _decrease_time_resolution(cochleagram, samplerate).flatten()
    padded = np.zeros((n_features,))
    padded[:flat.shape[0]] = flat

    prediction = model.predict([padded])[0].astype("int16")

    if file_names is not None:
        return file_names[prediction].split(".")[0]
    else:
        return prediction


def compute_model_accuracy(model, file_names, bg_file_names,
                           n_samples, n_features, use_mask, noise_level_range=(0.0, 1.0)):
    """Compute the model accuracy on random test data.

    This function generates random data for testing of the trained model. The data is either masked or unmasked,
    which will help to determine the usefulness of the masking. The accuracy score is a ratio of correct predictions
    to the overall number of predictions.

    :param Any model: The trained classifier
    :param list[str] file_names: Names of all available files in the dataset
    :param list[str] bg_file_names: List of names of files with background sounds
    :param int n_samples: Number of samples to generate for the testing set
    :param int n_features: Number of features (can't determine, because cochleagrams have different dimensions)
    :param bool use_mask: If True, the cochleagrams are masked before the prediction
    :param tuple noise_level_range: Low and high limits for `noise_level` (chosen uniformly from this interval)
    :returns: Accuracy score for the tested model
    :rtype: float

    """
    time_start = time.time()

    random_args = [
        {
            "label": random.choice(range(len(file_names))),
            "noise_level": random.uniform(noise_level_range[0], noise_level_range[1]),
            "use_white_noise": random.choice([True, False])
        }
        for _ in range(n_samples)
    ]

    score = accuracy_score(np.array([args["label"] for args in random_args], dtype="int16"),
                           np.array([make_prediction(
                               model,
                               n_features,
                               file_name=file_names[args["label"]],
                               use_mask=use_mask,
                               noise_level=args["noise_level"],
                               bg_file_name=(None
                                             if args["use_white_noise"]
                                             else random.choice(bg_file_names))
                           ) for args in random_args], dtype="int16"))

    time_end = time.time()
    print(f"Model accuracy on random data ({n_samples} samples) "
          + ("with" if use_mask else "without") + f" masking: {score:.2%}")
    print(f"Testing time: {time_end - time_start} s")

    return score
