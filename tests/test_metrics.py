"""Synthetic checks for metric values and the fixed activity label order."""

import unittest

import numpy as np

from har.metrics import calculate_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = calculate_metrics(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
        )

        for key in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ):
            self.assertAlmostEqual(result[key], 1.0)

        np.testing.assert_array_equal(
            result["confusion_matrix"],
            np.eye(6),
        )

    def test_one_expected_confusion(self):
        result = calculate_metrics(
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 5],
        )

        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertAlmostEqual(result["precision_macro"], 3 / 4)
        self.assertAlmostEqual(result["recall_macro"], 5 / 6)
        self.assertAlmostEqual(result["f1_macro"], 7 / 9)

        expected = np.eye(6, dtype=int)
        expected[5, 5] = 0
        expected[5, 4] = 1

        np.testing.assert_array_equal(
            result["confusion_matrix"],
            expected,
        )

    def test_missing_classes_preserve_six_class_order(self):
        result = calculate_metrics([1, 1], [1, 1])

        self.assertAlmostEqual(result["accuracy"], 1.0)

        for key in (
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ):
            self.assertAlmostEqual(result[key], 1 / 6)

        expected = np.zeros((6, 6), dtype=int)
        expected[0, 0] = 2

        np.testing.assert_array_equal(
            result["confusion_matrix"],
            expected,
        )

    def test_rejects_misaligned_predictions(self):
        with self.assertRaises(ValueError):
            calculate_metrics([1, 2], [1])