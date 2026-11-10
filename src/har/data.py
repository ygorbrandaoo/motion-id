"""Loader for the UCI HAR Dataset (calculated features, 561 columns).

Participant IDs are returned separately and must never be used as features.
"""

from pathlib import Path

import numpy as np

from har.metrics import ACTIVITY_LABELS
from har.validation import assert_disjoint_groups

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "UCI HAR Dataset"
)
N_FEATURES = 561
SPLITS = ("train", "test")


def _read(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path


def load_feature_names(data_dir=DEFAULT_DATA_DIR):
    """Return 561 unique names. features.txt repeats names, so the column
    index is prepended to every name (for example "303_tBodyAcc-...")."""
    lines = _read(Path(data_dir) / "features.txt").read_text().splitlines()
    names = []

    for line in lines:
        index, name = line.split(maxsplit=1)
        names.append(f"{index}_{name}")

    if len(names) != N_FEATURES or len(set(names)) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} unique feature names")

    return names


def load_activity_names(data_dir=DEFAULT_DATA_DIR):
    """Return {activity id: activity name} from activity_labels.txt."""
    lines = _read(Path(data_dir) / "activity_labels.txt").read_text().splitlines()
    names = {}

    for line in lines:
        index, name = line.split(maxsplit=1)
        names[int(index)] = name

    if tuple(sorted(names)) != tuple(ACTIVITY_LABELS):
        raise ValueError("activity_labels.txt must contain ids 1 through 6")

    return names


def load_split(split, data_dir=DEFAULT_DATA_DIR):
    """Return (X, y, groups) for "train" or "test".

    X has shape (n_windows, 561), y holds activity ids 1 to 6 and groups holds
    the participant id of each row. All three keep the original row order.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")

    folder = Path(data_dir) / split
    X = np.loadtxt(_read(folder / f"X_{split}.txt"))
    y = np.loadtxt(_read(folder / f"y_{split}.txt"), dtype=int)
    groups = np.loadtxt(_read(folder / f"subject_{split}.txt"), dtype=int)

    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X_{split}.txt must have {N_FEATURES} columns")

    if not len(X) == len(y) == len(groups):
        raise ValueError(f"{split} files have different numbers of rows")

    if not np.isin(y, ACTIVITY_LABELS).all():
        raise ValueError("y must use activity ids 1 through 6")

    if not np.isfinite(X).all():
        raise ValueError("X contains missing or infinite values")

    return X, y, groups


def load_dataset(data_dir=DEFAULT_DATA_DIR):
    """Load both official splits and check that no participant is in both."""
    X_train, y_train, g_train = load_split("train", data_dir)
    X_test, y_test, g_test = load_split("test", data_dir)

    assert_disjoint_groups(g_train, g_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "groups_train": g_train,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": g_test,
        "feature_names": load_feature_names(data_dir),
        "activity_names": load_activity_names(data_dir),
    }
