import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from har.evaluate import evaluate_grouped
from har.models.svm import SVM_PARAM_GRID, build_pipeline, main, tune_svm


def make_data(n_participants=8, rows_per_class=4, n_features=12, seed=0):
    """Small separable dataset: one cluster per activity, several participants."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []

    for participant in range(1, n_participants + 1):
        for label in range(1, 7):
            center = np.zeros(n_features)
            center[label - 1] = 4.0
            X.append(rng.normal(center, 1.0, size=(rows_per_class, n_features)))
            y += [label] * rows_per_class
            groups += [participant] * rows_per_class

    return np.vstack(X), np.array(y), np.array(groups)


def test_pipeline_scales_before_svm():
    pipeline = build_pipeline(random_state=7, C=10, kernel="rbf")

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["svm"], SVC)
    assert pipeline.named_steps["svm"].C == 10
    assert pipeline.named_steps["svm"].random_state == 7


def test_param_grid_targets_the_svm_step():
    for grid in SVM_PARAM_GRID:
        assert all(key.startswith("svm__") for key in grid)


def test_tune_svm_returns_unprefixed_best_params():
    X, y, groups = make_data()
    grid = [{"svm__kernel": ["linear"], "svm__C": [0.1, 10]}]

    result = tune_svm(X, y, groups, n_splits=4, param_grid=grid, n_jobs=1)

    assert set(result["best_params"]) == {"kernel", "C"}
    assert result["best_params"]["kernel"] == "linear"
    assert result["best_params"]["C"] in (0.1, 10)
    assert result["n_candidates"] == 2
    assert result["best_score"] > 0.5
    assert result["search_time_seconds"] >= 0


def test_tune_svm_rejects_more_splits_than_participants():
    X, y, groups = make_data(n_participants=3)

    with pytest.raises(ValueError):
        tune_svm(X, y, groups, n_splits=5, n_jobs=1)


def test_pipeline_works_with_the_shared_grouped_evaluation():
    X, y, groups = make_data()

    result = evaluate_grouped(
        build_pipeline(C=1, kernel="linear"), X, y, groups, model_name="svm", n_splits=4
    )

    assert result["model"] == "svm"
    assert len(result["folds"]) == 4
    assert result["summary"]["f1_macro"]["mean"] > 0.5


def test_holdout_requires_confirmation_flag():
    with pytest.raises(SystemExit):
        main(["--mode", "holdout"])