import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from har.feature_importance import calculate_importance


class FeatureImportanceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(60, 3))
        self.y = np.tile(np.arange(1, 7), 10)

    def test_exports_ranked_importances_and_png_or_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout_path = root / "holdout.json"
            holdout_path.write_text(
                json.dumps({
                    "best_params": {
                        "n_estimators": 8,
                        "max_features": "sqrt",
                    }
                }),
                encoding="utf-8",
            )
            config = {"random_state": 42, "data_dir": root}

            with (
                patch(
                    "har.feature_importance.load_config",
                    return_value=config,
                ),
                patch(
                    "har.feature_importance.load_split",
                    return_value=(self.X, self.y, np.ones(len(self.y))),
                ) as load_split,
                patch(
                    "har.feature_importance.load_feature_names",
                    return_value=["feature_a", "feature_b", "feature_c"],
                ),
            ):
                for suffix, signature in (
                    (".png", b"\x89PNG\r\n\x1a\n"),
                    (".pdf", b"%PDF"),
                ):
                    with self.subTest(format=suffix):
                        plot_path = root / f"importance{suffix}"
                        csv_path = root / f"importance{suffix}.csv"
                        ranked = calculate_importance(
                            holdout_result_path=holdout_path,
                            csv_path=csv_path,
                            plot_path=plot_path,
                            top_n=2,
                        )

                        with csv_path.open(
                            newline="",
                            encoding="utf-8",
                        ) as stream:
                            rows = list(csv.DictReader(stream))
                        report = json.loads(
                            csv_path.with_suffix(".json").read_text(
                                encoding="utf-8"
                            )
                        )

                        self.assertEqual(len(rows), 3)
                        self.assertEqual(
                            report["activity_names_pt"],
                            {
                                "1": "Caminhando",
                                "2": "Subindo",
                                "3": "Descendo escadas",
                                "4": "Sentado",
                                "5": "Em pé",
                                "6": "Deitado",
                            },
                        )
                        self.assertEqual(
                            len(report["feature_importances"]),
                            len(rows),
                        )
                        self.assertEqual(
                            [int(row["rank"]) for row in rows],
                            [1, 2, 3],
                        )
                        self.assertEqual(
                            [row["feature"] for row in rows],
                            [item["feature"] for item in ranked],
                        )
                        self.assertGreaterEqual(
                            float(rows[0]["importance"]),
                            float(rows[1]["importance"]),
                        )
                        self.assertTrue(
                            plot_path.read_bytes().startswith(signature)
                        )
                        if suffix == ".png":
                            png_data = plot_path.read_bytes()
                            phys_data_start = png_data.index(b"pHYs") + 4
                            pixels_per_meter = int.from_bytes(
                                png_data[phys_data_start:phys_data_start + 4],
                                "big",
                            )
                            self.assertAlmostEqual(
                                pixels_per_meter * 0.0254,
                                300,
                                delta=0.1,
                            )

            self.assertEqual(load_split.call_args.args[0], "train")

    def test_rejects_nonpositive_top_n(self):
        with self.assertRaisesRegex(ValueError, "top_n"):
            calculate_importance(top_n=0)


if __name__ == "__main__":
    unittest.main()
