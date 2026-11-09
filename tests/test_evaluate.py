"""Synthetic integration tests. Exports are written to temporary directories."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from har.evaluate import (
    evaluate_grouped,
    evaluate_holdout,
    export_comparison,
    save_results,
)


class FreshFitClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        if hasattr(self, "classes_"):
            raise AssertionError("Estimator was reused across folds")

        np.testing.assert_allclose(
            X.mean(axis=0),
            0,
            atol=1e-10,
        )

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(144, dtype=float).reshape(72, 2)
        self.y = np.tile(np.arange(1, 7), 12)
        self.groups = np.repeat(np.arange(1, 7), 12)

        self.model = make_pipeline(
            StandardScaler(),
            FreshFitClassifier(),
        )

    def evaluate(self):
        return evaluate_grouped(
            self.model,
            self.X,
            self.y,
            self.groups,
            model_name="synthetic_check",
            n_splits=3,
        )

    def test_fresh_pipeline_and_disjoint_participants(self):
        result = self.evaluate()

        self.assertFalse(
            hasattr(self.model.steps[-1][1], "classes_")
        )
        self.assertEqual(len(result["folds"]), 3)

        for fold in result["folds"]:
            self.assertFalse(
                set(fold["train_groups"])
                & set(fold["validation_groups"])
            )

            self.assertEqual(
                sum(map(sum, fold["confusion_matrix"])),
                fold["n_validation"],
            )

            self.assertGreaterEqual(fold["fit_time_seconds"], 0)
            self.assertGreaterEqual(fold["predict_time_seconds"], 0)

    def test_known_summary_and_sample_standard_deviation(self):
        groups = np.repeat(np.arange(1, 4), 24)
        labels = np.full(72, 2)

        for start, count in ((0, 2), (24, 4), (48, 6)):
            labels[start:start + count] = 1

        result = evaluate_grouped(
            self.model,
            self.X,
            labels,
            groups,
            model_name="synthetic_check",
            n_splits=3,
        )

        self.assertAlmostEqual(
            result["summary"]["accuracy"]["mean"],
            1 / 6,
        )
        self.assertAlmostEqual(
            result["summary"]["accuracy"]["std"],
            1 / 12,
        )

    def test_json_and_csv_exports(self):
        result = self.evaluate()

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "nested/result.json"
            csv_path = Path(directory) / "comparison.csv"

            save_results(result, json_path)
            export_comparison([result], csv_path)

            self.assertEqual(
                json.loads(json_path.read_text()),
                result,
            )

            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "synthetic_check")
            self.assertAlmostEqual(
                float(rows[0]["accuracy_mean"]),
                1 / 6,
            )

    def test_rejects_unknown_activity_labels(self):
        labels = self.y.copy()
        labels[0] = 7

        with self.assertRaisesRegex(ValueError, "activity labels"):
            evaluate_grouped(
                self.model,
                self.X,
                labels,
                self.groups,
                model_name="invalid",
            )

    def test_rejects_mixed_validation_protocols(self):
        first = self.evaluate()
        second = self.evaluate()
        second["n_splits"] = 5

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "same cross-validation protocol",
            ):
                export_comparison(
                    [first, second],
                    Path(directory) / "comparison.csv",
                )

    def test_rejects_different_participant_partitions(self):
        first = self.evaluate()
        second = self.evaluate()
        second["folds"][0]["validation_groups"] = [99]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "same cross-validation protocol",
            ):
                export_comparison(
                    [first, second],
                    Path(directory) / "comparison.csv",
                )

    def test_holdout_rejects_participant_overlap(self):
        with self.assertRaisesRegex(
            ValueError,
            "Participants occur in both",
        ):
            evaluate_holdout(
                self.model,
                self.X,
                self.y,
                self.groups,
                self.X,
                self.y,
                self.groups,
                model_name="invalid",
            )

    def test_holdout_report_is_separate_from_cross_validation(self):
        result = evaluate_holdout(
            self.model,
            self.X[:48],
            self.y[:48],
            self.groups[:48],
            self.X[48:],
            self.y[48:],
            self.groups[48:],
            model_name="synthetic_check",
        )

        self.assertEqual(result["evaluation"], "holdout")
        self.assertAlmostEqual(
            result["metrics"]["accuracy"],
            1 / 6,
        )
        self.assertEqual(
            sum(map(sum, result["metrics"]["confusion_matrix"])),
            24,
        )

        with tempfile.TemporaryDirectory() as directory:
            save_results(
                result,
                Path(directory) / "test.json",
            )

            with self.assertRaisesRegex(
                ValueError,
                "same cross-validation protocol",
            ):
                export_comparison(
                    [result],
                    Path(directory) / "comparison.csv",
                )