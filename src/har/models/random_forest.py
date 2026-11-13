"""Random Forest tuning and evaluation for the UCI HAR dataset.

Run from the repository root:

    python -m models.random_forest --mode cv
    python -m models.random_forest --mode holdout --confirm-frozen
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from har.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from har.data import load_split
from har.evaluate import evaluate_holdout, save_results, evaluate_grouped
from har.validation import group_kfold_splits

MODEL_NAME = "random_forest"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
CV_RESULT_PATH = RESULTS_DIR / "random_forest_cv.json"
HOLDOUT_RESULT_PATH = RESULTS_DIR / "random_forest_holdout.json"

#Esse PARAM_GRID é somente para testar se os resultados estão sendo salvos 
#corretamente, descomente esse para param_grid e comente o outro para testar
#o salvamento dos resultados!
#PARAM_GRID = {
#    "n_estimators": [2, 3],
#    "max_features": [None],
#    "max_depth": [2,3]
#}

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_features": ["sqrt", 0.05],
    "max_depth" : [None, 5, 10]
}
SELECTION_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)

def build_pipeline(random_state=42, **random_forest_params):
    pipeline = Pipeline( [
        ('scaler', StandardScaler()),
        ('random_forest', RandomForestClassifier(random_state=random_state, **random_forest_params))
    ])

    if random_forest_params:
        pipeline.set_params(**{f"random_forest__{key}": value for key, value in random_forest_params.items()})

    return pipeline

def _plain(value):
    return value.item() if isinstance(value, np.generic) else value

def run_grouped_search(
    X,
    y,
    groups,
    *,
    random_state=42,
    scoring="f1_macro",
    n_splits=5,
    n_jobs=1,
    param_grid=None,
):
    """Tune the forest using only folds that hold out complete participants."""
    if scoring not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRICS}"
        )

    splits = group_kfold_splits(X, y, groups, n_splits=n_splits)
    selected_param_grid = PARAM_GRID if param_grid is None else param_grid
    pipeline_param_grid = {
        (
            key if key.startswith("random_forest__")
            else f"random_forest__{key}"
        ): values
        for key, values in selected_param_grid.items()
    }
    search = GridSearchCV(
        estimator=build_pipeline(random_state),
        param_grid=pipeline_param_grid,
        scoring=scoring,
        cv=splits,
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
    )
    
    start = perf_counter()
    search.fit(X, y)
    search_seconds = perf_counter() - start

    results = search.cv_results_
    best = int(np.argmin(results["rank_test_score"]))
    return {
        "best_params": {
            key.removeprefix("random_forest__"): _plain(value)
            for key, value in results["params"][best].items()
        },
        "best_score": float(results["mean_test_score"][best]),
        "best_score_std": float(results["std_test_score"][best]),
        "scoring": scoring,
        "n_candidates": len(results["params"]),
        "n_splits": n_splits,
        "param_grid": selected_param_grid,
        "search_time_seconds": search_seconds
    }

def run_cv(
    config_path=DEFAULT_CONFIG,
    *,
    n_jobs=1,
    cv_result_path=CV_RESULT_PATH,
):
    """Tune on the official training partition without reading the test set."""
    config = load_config(config_path)
    X_train, y_train, train_groups = load_split("train", config["data_dir"])

    started = perf_counter()
    search = run_grouped_search(
        X_train,
        y_train,
        train_groups,
        random_state=config["random_state"],
        scoring=config["selection_metric"],
        n_splits=config["n_splits"],
        n_jobs=n_jobs,
    )

    estimator = build_pipeline(config["random_state"], **search["best_params"])
    search_report = evaluate_grouped(
        estimator,
        X_train,
        y_train,
        train_groups,
        model_name=MODEL_NAME,
        n_splits=config["n_splits"]
    )
    search_report["search"] = search

    save_results(search_report, cv_result_path)
    return search_report


def run_holdout(
    config_path=DEFAULT_CONFIG,
    *,
    cv_result_path=CV_RESULT_PATH,
    holdout_result_path=HOLDOUT_RESULT_PATH,
):
    """Evaluate the reserved test set using parameters saved by the CV stage."""
    config = load_config(config_path)
    if not cv_result_path.is_file():
        raise FileNotFoundError(f"Run --mode cv first; missing {cv_result_path}")

    best_params = json.loads(
        cv_result_path.read_text(encoding="utf-8")
    )["search"]["best_params"]
    X_train, y_train, train_groups = load_split("train", config["data_dir"])
    X_test, y_test, test_groups = load_split("test", config["data_dir"])

    evaluator = build_pipeline(config["random_state"], **best_params)
    holdout_report = evaluate_holdout(
        evaluator,
        X_train, y_train, train_groups,
        X_test, y_test, test_groups,
        model_name=MODEL_NAME,
    )
    holdout_report["best_params"] = best_params
    
    save_results(holdout_report, holdout_result_path)
    return holdout_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a Random Forest on UCI HAR."
    )
    parser.add_argument("--mode", choices=("cv", "holdout"), default="cv")
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="required for holdout: confirms the parameters were frozen",
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

    if args.mode == "holdout" and not args.confirm_frozen:
        parser.error(
            "holdout uses the official test set; pass --confirm-frozen "
            "after parameters are frozen"
        )

    if args.mode == "cv":
        search_report = run_cv(args.config, n_jobs=args.n_jobs)
        summary = search_report['summary']
        print(f"Best parameters: {search_report['search']['best_params']}")
        print(f"Search time: {search_report['search']['search_time_seconds']:.1f} s")
        for key in ("accuracy", "f1_macro", "fit_time_seconds"):
            print(f"CV {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")
        print(f"Saved: {CV_RESULT_PATH}")
    else:
        holdout_report = run_holdout(args.config)
        for key, value in holdout_report["metrics"].items():
            if key != "confusion_matrix":
                print(f"Test {key}: {value:.4f}")
        print(f"Saved: {HOLDOUT_RESULT_PATH}")

if __name__ == "__main__":
    main()
