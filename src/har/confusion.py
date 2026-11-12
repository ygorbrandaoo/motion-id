"""Confusion matrix and per-class error analysis for the final SVM (task T13).

Run only after T12: results/metrics/svm_holdout.json must exist. This module refits the
SVM with the frozen parameters saved in that file, predicts the official test partition
and writes tables and figures. Nothing is tuned here.

    PYTHONPATH=src uv run python -m har.confusion
    PYTHONPATH=src uv run --with matplotlib python -m har.confusion   # if matplotlib is missing
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from har.config import PROJECT_ROOT, load_config
from har.data import load_split
from har.evaluate import save_results
from har.metrics import ACTIVITY_LABELS
from har.models.svm import HOLDOUT_RESULT_PATH, RESULTS_DIR, build_pipeline

ACTIVITY_NAMES_PT = {
    1: "Caminhando",
    2: "Subindo escadas",
    3: "Descendo escadas",
    4: "Sentado",
    5: "Em pé",
    6: "Deitado",
}

CONFUSION_JSON = RESULTS_DIR / "svm_confusion.json"
PER_CLASS_CSV = RESULTS_DIR / "svm_per_class.csv"
PREDICTIONS_CSV = RESULTS_DIR / "svm_predictions.csv"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def analyze(y_true, y_pred):
    """Return the confusion matrices, per-class metrics and the most confused pairs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(ACTIVITY_LABELS)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros(matrix.shape, dtype=float),
        where=support[:, None] > 0,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "label": label,
            "name": ACTIVITY_NAMES_PT[label],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    ]

    confusions = [
        {
            "true": true,
            "true_name": ACTIVITY_NAMES_PT[true],
            "predicted": predicted,
            "predicted_name": ACTIVITY_NAMES_PT[predicted],
            "count": int(matrix[i, j]),
            "share_of_true_class": float(normalized[i, j]),
        }
        for i, true in enumerate(labels)
        for j, predicted in enumerate(labels)
        if i != j and matrix[i, j] > 0
    ]
    confusions.sort(key=lambda item: (-item["count"], item["true"], item["predicted"]))

    return {
        "labels": labels,
        "names": [ACTIVITY_NAMES_PT[label] for label in labels],
        "n_test": int(matrix.sum()),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
        "per_class": per_class,
        "top_confusions": confusions[:10],
    }


def plot_confusion(matrix, path_stem, *, normalized):
    """Save one confusion matrix as PNG (300 dpi) and PDF, sized for one IEEE column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    names = [ACTIVITY_NAMES_PT[label] for label in ACTIVITY_LABELS]

    fig, ax = plt.subplots(figsize=(3.46, 3.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)

    ticks = range(len(names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Atividade prevista", fontsize=7)
    ax.set_ylabel("Atividade real", fontsize=7)

    threshold = 0.5 if normalized else matrix.max() / 2
    for i in ticks:
        for j in ticks:
            value = matrix[i, j]
            text = f"{value:.2f}".replace(".", ",") if normalized else f"{int(value)}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=6,
                color="white" if value > threshold else "black",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=6)
    fig.tight_layout()

    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _save_tables(analysis, groups, y_true, y_pred):
    save_results(analysis, CONFUSION_JSON)

    with PER_CLASS_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["label", "name", "precision", "recall", "f1", "support"]
        )
        writer.writeheader()
        writer.writerows(analysis["per_class"])

    with PREDICTIONS_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", "participant", "y_true", "y_pred"])
        for row, (group, true, pred) in enumerate(zip(groups, y_true, y_pred)):
            writer.writerow([row, int(group), int(true), int(pred)])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Confusion analysis of the final SVM (T13).")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if not HOLDOUT_RESULT_PATH.is_file():
        raise SystemExit(
            f"Missing {HOLDOUT_RESULT_PATH}. Run T12 first "
            "(python -m har.models.svm --mode holdout --confirm-frozen)."
        )

    config = load_config()
    holdout = json.loads(HOLDOUT_RESULT_PATH.read_text(encoding="utf-8"))
    best_params = holdout["best_params"]

    X_train, y_train, _ = load_split("train", config["data_dir"])
    X_test, y_test, test_groups = load_split("test", config["data_dir"])

    model = build_pipeline(config["random_state"], **best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    expected = holdout["metrics"]["accuracy"]
    if abs(accuracy - expected) > 1e-9:
        raise SystemExit(
            f"Accuracy {accuracy:.6f} differs from svm_holdout.json ({expected:.6f}). "
            "Do not use these results; check data, parameters and library versions."
        )
    print(f"OK: reproduces the accuracy in svm_holdout.json ({accuracy:.4f}).")

    analysis = analyze(y_test, y_pred)
    _save_tables(analysis, test_groups, y_test, y_pred)

    print("\nPer-class results (test):")
    for item in analysis["per_class"]:
        print(
            f"  {item['name']:<17} precision {item['precision']:.3f}  "
            f"recall {item['recall']:.3f}  f1 {item['f1']:.3f}  n={item['support']}"
        )

    print("\nMost frequent confusions (true -> predicted):")
    for item in analysis["top_confusions"][:5]:
        print(
            f"  {item['true_name']} -> {item['predicted_name']}: "
            f"{item['count']} ({item['share_of_true_class']:.1%} of the true class)"
        )

    print(f"\nSaved: {CONFUSION_JSON}, {PER_CLASS_CSV}, {PREDICTIONS_CSV}")

    if args.no_figures:
        return

    try:
        plot_confusion(
            analysis["confusion_matrix_normalized"],
            args.figures_dir / "svm_confusion_normalized",
            normalized=True,
        )
        plot_confusion(
            analysis["confusion_matrix"],
            args.figures_dir / "svm_confusion_counts",
            normalized=False,
        )
    except ImportError:
        print(
            "\nmatplotlib is not installed, so the figures were skipped. Run again with:\n"
            "  PYTHONPATH=src uv run --with matplotlib python -m har.confusion"
        )
        return

    print(f"Figures saved in {args.figures_dir}")


if __name__ == "__main__":
    main()