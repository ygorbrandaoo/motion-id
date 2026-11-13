"""Extract and plot Random Forest feature importances."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from har.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from har.data import load_feature_names, load_split
from har.evaluate import save_results
from har.metrics import ACTIVITY_LABELS
from har.models.random_forest import (
    HOLDOUT_RESULT_PATH,
    RESULTS_DIR,
    build_pipeline,
)

ACTIVITY_NAMES_PT = {
    1: "Caminhando",
    2: "Subindo", 
    3: "Descendo escadas",
    4: "Sentado",
    5: "Em pé",
    6: "Deitado",
}

IMPORTANCE_CSV = RESULTS_DIR / "rf_feature_importances.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_PLOT_PATH = DEFAULT_FIGURES_DIR / "rf_feature_importances.png"


def calculate_importance(
    *,
    config_path=DEFAULT_CONFIG,
    holdout_result_path=HOLDOUT_RESULT_PATH,
    csv_path=IMPORTANCE_CSV,
    json_path=None,
    plot_path: Path | None = DEFAULT_PLOT_PATH,
    top_n=20,
):
    """Fit the holdout-configured forest on training data and export importances."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    holdout_result_path = Path(holdout_result_path)
    if not holdout_result_path.is_file():
        raise FileNotFoundError(
            f"Missing holdout results: {holdout_result_path}"
        )

    holdout_report = json.loads(
        holdout_result_path.read_text(encoding="utf-8")
    )
    best_params = holdout_report["best_params"]
    config = load_config(config_path)
    X_train, y_train, _ = load_split("train", config["data_dir"])
    feature_names = load_feature_names(config["data_dir"])

    forest = build_pipeline(config["random_state"], **best_params)
    forest.fit(X_train, y_train)
    importances = forest.named_steps["random_forest"].feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "The number of feature names does not match the fitted forest"
        )

    order = np.argsort(-importances, kind="stable")
    ranked = [
        {
            "rank": rank,
            "feature": feature_names[index],
            "importance": float(importances[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["rank", "feature", "importance"],
        )
        writer.writeheader()
        writer.writerows(ranked)

    if json_path is None:
        json_path = csv_path.with_suffix(".json")
    save_results(
        {
            "activity_names_pt": {
                str(label): ACTIVITY_NAMES_PT[label]
                for label in ACTIVITY_LABELS
            },
            "feature_importances": ranked,
        },
        json_path,
    )

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_format = plot_path.suffix.lower().lstrip(".")
        if plot_format not in {"png", "pdf"}:
            raise ValueError("plot_path must end in .png or .pdf")

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        shown = ranked[:top_n]
        figure_height = max(4, 0.3 * len(shown))
        figure, axis = plt.subplots(figsize=(10, figure_height))
        axis.barh(
            [item["feature"] for item in reversed(shown)],
            [item["importance"] for item in reversed(shown)],
        )
        axis.set_xlabel("Mean decrease in impurity")
        axis.set_title(f"Top {len(shown)} Random Forest feature importances")
        figure.tight_layout()
        if plot_format == "png":
            figure.savefig(plot_path, format=plot_format, dpi=300)
        else:
            figure.savefig(plot_path, format=plot_format)
        plt.close(figure)

    return ranked


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract feature importances from the holdout Random Forest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--holdout-result",
        type=Path,
        default=HOLDOUT_RESULT_PATH,
    )
    parser.add_argument("--csv", type=Path, default=IMPORTANCE_CSV)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf"),
        default="png",
        help="PNG is saved at 300 dpi; PDF is saved as vector output.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    plot_path = (
        None
        if args.no_figures
        else args.figures_dir / f"rf_feature_importances.{args.plot_format}"
    )
    ranked = calculate_importance(
        config_path=args.config,
        holdout_result_path=args.holdout_result,
        csv_path=args.csv,
        plot_path=plot_path,
        top_n=args.top_n,
    )
    print(f"Saved importances: {args.csv}")
    print(f"Saved activity-name mapping and importances: {args.csv.with_suffix('.json')}")
    if plot_path is not None:
        print(f"Saved plot: {plot_path}")
    print("Activity labels (Portuguese):")
    for label in ACTIVITY_LABELS:
        print(f"  {label}: {ACTIVITY_NAMES_PT[label]}")
    print("Top features:")
    for item in ranked[:args.top_n]:
        print(f"{item['rank']:>4}. {item['feature']}: {item['importance']:.6f}")


if __name__ == "__main__":
    main()
