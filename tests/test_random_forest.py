"""Synthetic tests for Random Forest model construction and grouped tuning."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.pipeline import Pipeline

from har.validation import group_kfold_splits
from har.models.random_forest import build_pipeline, run_grouped_search, run_holdout


class RandomForestTests(unittest.TestCase):
    def test_pipeline_is_unfitted(self):
        model = build_pipeline(random_state=17)
        self.assertIsInstance(model, Pipeline)
        self.assertFalse(hasattr(model.named_steps["random_forest"], "estimators_"))

    def test_search_keeps_participants_in_one_fold_partition(self):
        X = np.arange(72, dtype=float).reshape(36, 2)
        y = np.tile(np.arange(1, 7), 6)
        groups = np.repeat(np.arange(1, 7), 6)
        splits = group_kfold_splits(X, y, groups, n_splits=3)

        with patch(
            "har.models.random_forest.group_kfold_splits",
            return_value=splits,
        ):
            search = run_grouped_search(
                X,
                y,
                groups,
                n_splits=3,
                param_grid={"n_estimators": [5], "max_features": ["sqrt"]},
            )

        validation_count = np.zeros(len(groups), dtype=int)
        for train_indices, validation_indices in splits:
            train_groups = set(groups[train_indices])
            validation_groups = set(groups[validation_indices])
            self.assertFalse(train_groups & validation_groups)
            validation_count[validation_indices] += 1

        np.testing.assert_array_equal(validation_count, np.ones(len(groups)))
        self.assertEqual(search["best_params"]["n_estimators"], 5)
        self.assertNotIn("random_forest__n_estimators", search["best_params"])
        self.assertEqual(search["n_splits"], 3)

    def test_search_rejects_non_metric_selection(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            run_grouped_search(
                np.ones((12, 2)),
                np.tile(np.arange(1, 7), 2),
                np.repeat(np.arange(1, 7), 2),
                scoring="fit_time_seconds",
                n_splits=3,
            )

    def test_holdout_uses_saved_frozen_parameters(self):
        frozen_params = {
            "n_estimators": 7,
            "max_features": "sqrt",
            "max_depth": 5,
        }

        with TemporaryDirectory() as directory:
            cv_path = Path(directory) / "random_forest_cv.json"
            cv_path.write_text(
                json.dumps({"search": {"best_params": frozen_params}}),
                encoding="utf-8",
            )
            config = {"random_state": 17, "data_dir": Path(directory)}
            train_data = (
                np.ones((2, 2)),
                np.array([1, 2]),
                np.array([1, 2]),
            )
            test_data = (
                np.ones((1, 2)),
                np.array([3]),
                np.array([3]),
            )

            with (
                patch("har.models.random_forest.load_config", return_value=config),
                patch(
                    "har.models.random_forest.load_split",
                    side_effect=[train_data, test_data],
                ) as load_split,
                patch(
                    "har.models.random_forest.evaluate_holdout",
                    return_value={"evaluation": "holdout", "metrics": {}},
                ) as evaluate_holdout,
                patch("har.models.random_forest.save_results"),
            ):
                result = run_holdout(
                    cv_result_path=cv_path,
                    holdout_result_path=Path(directory) / "holdout.json",
                )

        estimator = evaluate_holdout.call_args.args[0]
        self.assertEqual(
            estimator.get_params()["random_forest__n_estimators"],
            frozen_params["n_estimators"],
        )
        self.assertEqual(result["best_params"], frozen_params)
        self.assertEqual(
            [call.args[0] for call in load_split.call_args_list],
            ["train", "test"],
        )


if __name__ == "__main__":
    unittest.main()
