"""Random Forest tuning and evaluation for the UCI HAR dataset."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from har.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from har.data import load_split
from har.evaluate import evaluate_holdout, save_results
from har.validation import group_kfold_splits

MODEL_NAME = "random_forest"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
CV_RESULT_PATH = RESULTS_DIR / "random_forest_cv.json"
HOLDOUT_RESULT_PATH = RESULTS_DIR / "random_forest_holdout.json"
PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_features": ["sqrt", 0.5],
}
SELECTION_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)


def build_estimator(random_state=42):
    """Create an unfitted, reproducible Random Forest classifier."""
    return RandomForestClassifier(random_state=random_state)


def run_grouped_search(
    X,
    y,
    groups,
    *,
    random_state=42,
    selection_metric="f1_macro",
    n_splits=5,
    n_jobs=1,
    param_grid=None,
):
    """Tune the forest using only folds that hold out complete participants."""
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRICS}"
        )

    splits = group_kfold_splits(X, y, groups, n_splits=n_splits)
    search = GridSearchCV(
        estimator=build_estimator(random_state),
        param_grid=PARAM_GRID if param_grid is None else param_grid,
        scoring=selection_metric,
        cv=splits,
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def _search_report(
    search,
    groups,
    *,
    selection_metric,
    n_splits,
    elapsed_seconds,
):
    """Convert GridSearchCV's results to JSON-serializable values."""
    return {
        "model": MODEL_NAME,
        "evaluation": "grouped_hyperparameter_search",
        "selection_metric": selection_metric,
        "n_splits": n_splits,
        "elapsed_seconds": elapsed_seconds,
        "folds": [
            {
                "fold": fold_number,
                "n_train": len(train_indices),
                "n_validation": len(validation_indices),
                "train_groups": sorted(set(groups[train_indices].tolist())),
                "validation_groups": sorted(
                    set(groups[validation_indices].tolist())
                ),
            }
            for fold_number, (train_indices, validation_indices) in enumerate(
                search.cv,
                start=1,
            )
        ],
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "candidates": [
            {
                "params": params,
                "mean_test_score": float(mean_score),
                "std_test_score": float(std_score),
            }
            for params, mean_score, std_score in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
                strict=True,
            )
        ],
    }


def run_experiment(
    config_path=DEFAULT_CONFIG,
    *,
    n_jobs=1,
    cv_result_path=CV_RESULT_PATH,
    holdout_result_path=HOLDOUT_RESULT_PATH,
):
    """Tune on the official training partition, then evaluate the reserved test set."""
    config = load_config(config_path)
    X_train, y_train, train_groups = load_split("train", config["data_dir"])

    started = perf_counter()
    search = run_grouped_search(
        X_train,
        y_train,
        train_groups,
        random_state=config["random_state"],
        selection_metric=config["selection_metric"],
        n_splits=config["n_splits"],
        n_jobs=n_jobs,
    )
    search_report = _search_report(
        search,
        train_groups,
        selection_metric=config["selection_metric"],
        n_splits=config["n_splits"],
        elapsed_seconds=perf_counter() - started,
    )
    save_results(search_report, cv_result_path)

    X_test, y_test, test_groups = load_split("test", config["data_dir"])
    holdout_report = evaluate_holdout(
        search.best_estimator_,
        X_train,
        y_train,
        train_groups,
        X_test,
        y_test,
        test_groups,
        model_name=MODEL_NAME,
    )
    save_results(holdout_report, holdout_result_path)
    return search_report, holdout_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a Random Forest on UCI HAR."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel GridSearchCV jobs (default: 1).",
    )
    args = parser.parse_args(argv)

    search_report, holdout_report = run_experiment(
        args.config,
        n_jobs=args.n_jobs,
    )
    print(json.dumps({
        "best_params": search_report["best_params"],
        "best_score": search_report["best_score"],
        "holdout_metrics": holdout_report["metrics"],
        "cv_result_path": str(CV_RESULT_PATH),
        "holdout_result_path": str(HOLDOUT_RESULT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
