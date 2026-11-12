"""Synthetic tests for Random Forest model construction and grouped tuning."""

import unittest

import numpy as np

from models.random_forest import build_estimator, run_grouped_search


class RandomForestTests(unittest.TestCase):
    def test_estimator_uses_requested_random_state(self):
        model = build_estimator(random_state=17)
        self.assertFalse(hasattr(model, "estimators_"))

    def test_search_keeps_participants_in_one_fold_partition(self):
        X = np.arange(72, dtype=float).reshape(36, 2)
        y = np.tile(np.arange(1, 7), 6)
        groups = np.repeat(np.arange(1, 7), 6)

        search = run_grouped_search(
            X,
            y,
            groups,
            n_splits=3,
            param_grid={"n_estimators": [5], "max_features": ["sqrt"]},
        )

        validation_count = np.zeros(len(groups), dtype=int)
        for train_indices, validation_indices in getattr(search, "cv"):
            train_groups = set(groups[train_indices])
            validation_groups = set(groups[validation_indices])
            self.assertFalse(train_groups & validation_groups)
            validation_count[validation_indices] += 1

        np.testing.assert_array_equal(validation_count, np.ones(len(groups)))
        self.assertEqual(search.best_params_["n_estimators"], 5)

    def test_search_rejects_non_metric_selection(self):
        with self.assertRaisesRegex(ValueError, "selection_metric"):
            run_grouped_search(
                np.ones((12, 2)),
                np.tile(np.arange(1, 7), 2),
                np.repeat(np.arange(1, 7), 2),
                selection_metric="fit_time_seconds",
                n_splits=3,
            )


if __name__ == "__main__":
    unittest.main()
