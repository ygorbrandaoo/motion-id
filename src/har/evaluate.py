"""Evaluation functions for the UCI HAR datasets."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.utils.validation import check_consistent_length

from har.metrics import ACTIVITY_LABELS, calculate_metrics
from har.validation import assert_disjoint_groups, group_kfold_splits


SCALAR_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "fit_time_seconds",
    "predict_time_seconds",
)


def _arrays(X, y, groups):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)

    if X.ndim != 2 or y.ndim != 1 or groups.ndim != 1:
        raise ValueError("X must be 2D; y and groups must be 1D")

    check_consistent_length(X, y, groups)

    if len(y) == 0 or any(g is None or g != g for g in groups):
        raise ValueError(
            "Inputs must be nonempty with no missing participant IDs"
        )

    if not np.isin(y, ACTIVITY_LABELS).all():
        raise ValueError("y must use activity labels 1 through 6")

    return X, y, groups


def _fit_and_score(estimator, X_train, y_train, X_eval, y_eval):
    model = clone(estimator)

    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    prediction = model.predict(X_eval)
    predict_seconds = perf_counter() - start

    scores = calculate_metrics(y_eval, prediction)

    return {
        **{
            key: float(scores[key])
            for key in SCALAR_METRICS[:4]
        },
        "confusion_matrix": scores["confusion_matrix"].tolist(),
        "fit_time_seconds": fit_seconds,
        "predict_time_seconds": predict_seconds,
    }


def _metadata(estimator, model_name):
    return {
        "model": model_name,
        "activity_labels": list(ACTIVITY_LABELS),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parameters": {
            key: repr(value)
            for key, value in estimator.get_params().items()
        },
    }


def evaluate_grouped(estimator, X, y, groups, *, model_name, n_splits=5):
    """Use training data only; fit preprocessing inside the estimator Pipeline."""
    X, y, groups = _arrays(X, y, groups)

    splits = group_kfold_splits(X, y, groups, n_splits=n_splits)

    folds = []

    for fold, (train, validation) in enumerate(splits, start=1):
        X_train, X_validation = X[train], X[validation]
        y_train, y_validation = y[train], y[validation]

        folds.append({
            "fold": fold,
            "n_train": len(train),
            "n_validation": len(validation),
            "train_groups": np.unique(groups[train]).tolist(),
            "validation_groups": np.unique(groups[validation]).tolist(),
            "validation_class_counts": {
                str(label): int(np.sum(y_validation == label))
                for label in ACTIVITY_LABELS
            },
            **_fit_and_score(estimator, X_train, y_train, X_validation, y_validation,),
        })

    summary = {
        key: {
            "mean": float(np.mean([fold[key] for fold in folds])),
            "std": float(
                np.std([fold[key] for fold in folds], ddof=1)
            ),
        }
        for key in SCALAR_METRICS
    }

    return {
        **_metadata(estimator, model_name),
        "evaluation": "grouped_cross_validation",
        "n_splits": n_splits,
        "std_ddof": 1,
        "folds": folds,
        "summary": summary,
    }


def evaluate_holdout(estimator, X_train, y_train, train_groups, X_test, y_test, test_groups, *, model_name,):
    """Run only after tuning: fit on training data and score the reserved test set."""
    X_train, y_train, train_groups = _arrays(
        X_train,
        y_train,
        train_groups,
    )

    X_test, y_test, test_groups = _arrays(X_test, y_test, test_groups)

    assert_disjoint_groups(train_groups, test_groups)

    return {
        **_metadata(estimator, model_name),
        "evaluation": "holdout",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_groups": np.unique(train_groups).tolist(),
        "test_groups": np.unique(test_groups).tolist(),
        "metrics": _fit_and_score(estimator, X_train, y_train, X_test, y_test),
    }

def save_results(result, path):
    """Save a complete validation or final test report as JSON."""
    content = json.dumps(
        result,
        indent=2,
        allow_nan=False,
    ) + "\n"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def export_comparison(results, path):
    """Export CV means/stds; callers must use the same data and fold protocol."""
    results = list(results)

    if not results:
        raise ValueError("At least one result is required")

    if any(
        result["evaluation"] != "grouped_cross_validation"
        for result in results
    ):
        raise ValueError(
            "Results must use the same cross-validation protocol"
        )

    def protocol(result):
        return (
            result["evaluation"],
            result["n_splits"],
            tuple(
                (
                    tuple(fold["validation_groups"]),
                    fold["n_train"],
                    fold["n_validation"],
                )
                for fold in result["folds"]
            ),
        )

    if any(
        protocol(result) != protocol(results[0])
        for result in results
    ):
        raise ValueError(
            "Results must use the same cross-validation protocol"
        )

    columns = ["model", "n_splits"] + [
        f"{key}_{stat}"
        for key in SCALAR_METRICS
        for stat in ("mean", "std")
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "model": result["model"],
                "n_splits": result["n_splits"],
                **{
                    f"{key}_{stat}": result["summary"][key][stat]
                    for key in SCALAR_METRICS
                    for stat in ("mean", "std")
                },
            })