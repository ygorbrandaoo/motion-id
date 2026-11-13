"""Participant-aware cross-validation for the UCI HAR training partition.
Pass the official training partition only. Kepp the official teste parttition untouched until the model selection and
tuning have finished.
"""

from __future__ import annotations

from typing  import Any

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import  check_consistent_length

def assert_disjoint_groups(train_groups: Any, validation_groups: Any) -> None:
    """Raise ValueError of participant IDs occur in both partitions.
    This  also work for chequing the official train/test participant split.
    """
    shared = np.intersect1d(np.asarray(train_groups), np.asarray(validation_groups))

    if shared.size:
        raise ValueError(f"Participants occur in both partitions: {shared.tolist()}")

def group_kfold_splits(X: Any, y: Any, groups: Any, *, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_indices, validation_indices) for each GroupKFold fold.
    X, y and groups must contain rows in the same order. 
    Groups contains participant IDs and must never be included among model features.
    Index arrays refer to rows in the supplied training partition.
    The default GroupKFold ordering is deterministic (no shuffling).
    """

    if  groups is None:
        raise ValueError("groups must contain one participant ID per row")

    if (isinstance(n_splits, bool) or not  isinstance(n_splits, int) or n_splits < 2):
        raise ValueError("n_splits must be an interger of at last 2")

    group_ids = np.asarray(groups)

    if group_ids.ndim != 1:
        raise ValueError("groups must be one-dimensional")

    check_consistent_length(X, y, group_ids)

    if any (group is None or group != group for group in group_ids):
        raise ValueError("groups cannot contain missing participants IDs")

    if np.unique(group_ids).size < n_splits:
        raise ValueError("n_splits cannot exceed the number of participants")

    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=group_ids))

    for train_indices, validation_indices in folds:
        assert_disjoint_groups(group_ids[train_indices], group_ids[validation_indices])

    return folds