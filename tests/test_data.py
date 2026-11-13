"""Loader tests. The first class uses a tiny fake dataset; the second runs on
the real files and is skipped when they are not present locally."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from har.data import (
    DEFAULT_DATA_DIR,
    N_FEATURES,
    load_activity_names,
    load_dataset,
    load_feature_names,
    load_split,
)

ACTIVITIES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


def write_fake(root, *, n_columns=N_FEATURES, test_subjects=(3, 4)):
    root = Path(root)
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()

    # A repeated name, as in the real features.txt.
    names = [f"{i + 1} f{i % 10}" for i in range(N_FEATURES)]
    (root / "features.txt").write_text("\n".join(names) + "\n")
    (root / "activity_labels.txt").write_text(
        "\n".join(f"{i + 1} {n}" for i, n in enumerate(ACTIVITIES)) + "\n"
    )

    for split, subjects in (("train", (1, 2)), ("test", test_subjects)):
        n = 6 * len(subjects)
        rng = np.random.default_rng(0)
        np.savetxt(root / split / f"X_{split}.txt", rng.normal(size=(n, n_columns)))
        np.savetxt(root / split / f"y_{split}.txt", np.tile(np.arange(1, 7), len(subjects)), fmt="%d")
        np.savetxt(root / split / f"subject_{split}.txt", np.repeat(subjects, 6), fmt="%d")


class FakeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "ds"

    def test_feature_names_are_unique(self):
        write_fake(self.root)
        names = load_feature_names(self.root)
        self.assertEqual(len(names), N_FEATURES)
        self.assertEqual(len(set(names)), N_FEATURES)

    def test_activity_names(self):
        write_fake(self.root)
        self.assertEqual(load_activity_names(self.root)[4], "SITTING")

    def test_load_split_shapes_and_order(self):
        write_fake(self.root)
        X, y, groups = load_split("train", self.root)
        self.assertEqual(X.shape, (12, N_FEATURES))
        self.assertEqual(y.tolist(), [1, 2, 3, 4, 5, 6] * 2)
        self.assertEqual(groups.tolist(), [1] * 6 + [2] * 6)

    def test_load_dataset_keys(self):
        write_fake(self.root)
        data = load_dataset(self.root)
        self.assertEqual(set(data["groups_train"]), {1, 2})
        self.assertEqual(set(data["groups_test"]), {3, 4})

    def test_overlapping_participants_are_rejected(self):
        write_fake(self.root, test_subjects=(2, 4))
        with self.assertRaises(ValueError):
            load_dataset(self.root)

    def test_wrong_number_of_columns_is_rejected(self):
        write_fake(self.root, n_columns=10)
        with self.assertRaises(ValueError):
            load_split("train", self.root)

    def test_missing_file_is_reported(self):
        write_fake(self.root)
        (self.root / "train" / "y_train.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            load_split("train", self.root)

    def test_invalid_split_name(self):
        with self.assertRaises(ValueError):
            load_split("validation", self.root)


@unittest.skipUnless(DEFAULT_DATA_DIR.is_dir(), "real dataset not found locally")
class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_official_sizes(self):
        d = self.data
        self.assertEqual(d["X_train"].shape, (7352, 561))
        self.assertEqual(d["X_test"].shape, (2947, 561))

    def test_participants(self):
        d = self.data
        self.assertEqual(len(np.unique(d["groups_train"])), 21)
        self.assertEqual(len(np.unique(d["groups_test"])), 9)

    def test_all_six_activities_in_both_splits(self):
        d = self.data
        self.assertEqual(set(d["y_train"]), set(range(1, 7)))
        self.assertEqual(set(d["y_test"]), set(range(1, 7)))

    def test_feature_names_unique(self):
        self.assertEqual(len(set(self.data["feature_names"])), 561)


if __name__ == "__main__":
    unittest.main()
