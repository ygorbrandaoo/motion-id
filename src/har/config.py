"""Shared experiment settings, read from configs/default.yaml."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
REQUIRED_KEYS = ("data_dir", "n_splits", "random_state", "selection_metric")


def load_config(path=DEFAULT_CONFIG):
    """Return the settings as a dict; data_dir is resolved against the project root."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    config["data_dir"] = PROJECT_ROOT / config["data_dir"]
    return config
