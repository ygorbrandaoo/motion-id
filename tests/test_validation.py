"""Synthetic tests. No dataset download or model trainingrequired."""

import unittest
import numpy as np

from har.validation import (assert_disjoint_groups, group_kfold_splits)

class GroupValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(24).reshape(12, 2)
        self.y = np.array([1, 2] * 6)
        self.groups = np.repeat(np.arange(1, 7), 2)

    def test_each_fold_keeps_participants_separate(self):
        folds = group_kfold_splits(self.X, self.y, self.groups, n_splits=3)

        self.assertEqual(len(folds), 3)
        held_out = np.zeros(len(self.groups), dtype=int)

        for train, validation in folds:
            self.assertFalse(set(train) & set(validation))

            self.assertEqual(len(train) + len(validation), len(self.groups))

            assert_disjoint_groups(self.groups[train], self.groups[validation])

            held_out[validation] += 1

        np.testing.assert_array_equal(held_out, np.ones(len(self.groups)))

    def test_rejects_overlap_in_official_partitions(self):
        with self.assertRaisesRegex(ValueError, "Participants occur in both"):
            assert_disjoint_groups([1, 2, 3], [3, 4])

    def test_rejects_misaligned_rows(self):
        with self.assertRaises(ValueError):
            group_kfold_splits(self.X, self.y[:-1], self.groups)

    def test_rejects_too_few_participants(self):
        with self.assertRaisesRegex(ValueError, "number of participants"):
            group_kfold_splits(self.X, self.y, self.groups, n_splits=7)

    def test_rejects_missing_participant(self):
        groups = self.groups.astype(float)
        groups[0] = np.nan

        with self.assertRaisesRegex(ValueError, "missing participant"):
            group_kfold_splits(self.X, self.y, groups)

if __name__ == "__main__":
    unittest.main()