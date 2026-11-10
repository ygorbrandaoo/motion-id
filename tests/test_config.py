import tempfile
import unittest
from pathlib import Path

from har.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_matches_agreed_protocol(self):
        config = load_config()
        self.assertEqual(config["n_splits"], 5)
        self.assertEqual(config["selection_metric"], "f1_macro")
        self.assertTrue(config["data_dir"].is_absolute())

    def test_missing_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("n_splits: 5\n")
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
