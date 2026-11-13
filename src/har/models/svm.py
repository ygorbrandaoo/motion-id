"""SVM classifier for the UCI HAR dataset (task T08).

Builds a StandardScaler + SVC pipeline, tunes it with participant-grouped
cross-validation on the official training partition and, only after the
hyperparameters are frozen (T11), scores it once on the official test partition.

Run from the repository root:

    uv run python -m har.models.svm --mode cv
    uv run python -m har.models.svm --mode holdout --confirm-frozen   # T12 only
"""

import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from har.config import PROJECT_ROOT, load_config
from har.data import load_split
from har.evaluate import evaluate_grouped, evaluate_holdout, save_results
from har.validation import group_kfold_splits

MODEL_NAME = "svm"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
CV_RESULT_PATH = RESULTS_DIR / "svm_cv.json"
HOLDOUT_RESULT_PATH = RESULTS_DIR / "svm_holdout.json"

# Small grid on purpose (C, kernel, gamma). gamma only exists for the RBF kernel,
# so the grid is a list of two dicts. Keys use the "svm__" Pipeline prefix.
SVM_PARAM_GRID = [
    {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1]},
    {
        "svm__kernel": ["rbf"],
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01],
    },
]


def build_pipeline(random_state=42, **svm_params):
    """Return StandardScaler + SVC.

    The scaler lives inside the Pipeline, so every fit (each CV fold and the final
    fit) learns its mean and variance from training rows only.
    svm_params are unprefixed SVC arguments, for example C=10, kernel="rbf".
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=random_state, cache_size=500)),
    ])

    if svm_params:
        pipeline.set_params(**{f"svm__{key}": value for key, value in svm_params.items()})

    return pipeline


def _plain(value):
    """Convert numpy scalars to Python types so the result is JSON serializable."""
    return value.item() if isinstance(value, np.generic) else value


def tune_svm(
    X,
    y,
    groups,
    *,
    n_splits=5,
    random_state=42,
    scoring="f1_macro",
    param_grid=None,
    n_jobs=-1,
):
    """Grid search with participant-grouped folds on the training partition only.

    Returns the best unprefixed parameters plus the search metadata. The pipeline is
    refit later (CV report and final test), so the search itself does not refit.
    Ties keep the first candidate in grid order, which keeps the choice deterministic.
    """
    param_grid = SVM_PARAM_GRID if param_grid is None else param_grid
    splits = group_kfold_splits(X, y, groups, n_splits=n_splits)

    search = GridSearchCV(
        build_pipeline(random_state),
        param_grid,
        scoring=scoring,
        cv=splits,
        n_jobs=n_jobs,
        refit=False,
        error_score="raise",
    )

    start = perf_counter()
    search.fit(X, y)
    search_seconds = perf_counter() - start

    results = search.cv_results_
    best = int(np.argmin(results["rank_test_score"]))

    return {
        "best_params": {
            key.removeprefix("svm__"): _plain(value)
            for key, value in results["params"][best].items()
        },
        "best_score": float(results["mean_test_score"][best]),
        "best_score_std": float(results["std_test_score"][best]),
        "scoring": scoring,
        "n_candidates": len(results["params"]),
        "n_splits": n_splits,
        "param_grid": param_grid,
        "search_time_seconds": search_seconds,
    }


def run_cv(config=None, path=CV_RESULT_PATH):
    """Tune on the training partition, then report grouped CV for the best setting.

    The official test files are not read in this mode.
    """
    config = load_config() if config is None else config
    X, y, groups = load_split("train", config["data_dir"])

    search = tune_svm(
        X,
        y,
        groups,
        n_splits=config["n_splits"],
        random_state=config["random_state"],
        scoring=config["selection_metric"],
    )

    estimator = build_pipeline(config["random_state"], **search["best_params"])
    result = evaluate_grouped(
        estimator, X, y, groups, model_name=MODEL_NAME, n_splits=config["n_splits"]
    )
    result["search"] = search

    save_results(result, path)
    return result


def run_holdout(config=None, cv_path=CV_RESULT_PATH, path=HOLDOUT_RESULT_PATH):
    """Final evaluation on the official test partition (T12), using frozen parameters.

    The parameters come from the saved CV result, never from the test set.
    """
    config = load_config() if config is None else config

    if not cv_path.is_file():
        raise FileNotFoundError(f"Run --mode cv first; missing {cv_path}")

    best_params = json.loads(cv_path.read_text(encoding="utf-8"))["search"]["best_params"]

    X_train, y_train, train_groups = load_split("train", config["data_dir"])
    X_test, y_test, test_groups = load_split("test", config["data_dir"])

    estimator = build_pipeline(config["random_state"], **best_params)
    result = evaluate_holdout(
        estimator,
        X_train, y_train, train_groups,
        X_test, y_test, test_groups,
        model_name=MODEL_NAME,
    )
    result["best_params"] = best_params

    save_results(result, path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune and evaluate the SVM (T08).")
    parser.add_argument("--mode", choices=("cv", "holdout"), default="cv")
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="required for holdout: confirms the parameters were frozen (T11)",
    )
    args = parser.parse_args(argv)

    if args.mode == "holdout" and not args.confirm_frozen:
        parser.error("holdout uses the official test set; pass --confirm-frozen after T11")

    if args.mode == "cv":
        result = run_cv()
        summary = result["summary"]
        print(f"Best parameters: {result['search']['best_params']}")
        print(f"Search time: {result['search']['search_time_seconds']:.1f} s")
        for key in ("accuracy", "f1_macro", "fit_time_seconds"):
            print(f"CV {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
        print(f"Saved: {CV_RESULT_PATH}")
    else:
        result = run_holdout()
        for key, value in result["metrics"].items():
            if key != "confusion_matrix":
                print(f"Test {key}: {value:.4f}")
        print(f"Saved: {HOLDOUT_RESULT_PATH}")


if __name__ == "__main__":
    main()