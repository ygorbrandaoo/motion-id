import numpy as np
import pytest

from har.confusion import analyze, plot_confusion


def sample():
    y_true = np.array([1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])
    y_pred = np.array([1, 1, 1, 2, 2, 2, 3, 2, 4, 5, 4, 4, 5, 5, 6, 6])
    return y_true, y_pred


def test_analyze_confusion_matrix_and_rows():
    result = analyze(*sample())
    matrix = np.array(result["confusion_matrix"])

    assert matrix.shape == (6, 6)
    assert matrix.sum() == result["n_test"] == 16
    assert matrix[0, 1] == 1  # one "Caminhando" predicted as "Subindo escadas"
    assert np.allclose(np.array(result["confusion_matrix_normalized"]).sum(axis=1), 1)


def test_analyze_per_class_metrics():
    result = analyze(*sample())
    walking, upstairs = result["per_class"][0], result["per_class"][1]

    assert walking["support"] == 4
    assert walking["recall"] == pytest.approx(0.75)
    assert upstairs["precision"] == pytest.approx(0.5)


def test_top_confusions_sorted_and_off_diagonal_only():
    confusions = analyze(*sample())["top_confusions"]

    assert all(item["true"] != item["predicted"] for item in confusions)
    assert [item["count"] for item in confusions] == sorted(
        (item["count"] for item in confusions), reverse=True
    )
    assert confusions[0]["true"] == 1 and confusions[0]["predicted"] == 2


def test_plot_writes_png_and_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    result = analyze(*sample())

    plot_confusion(result["confusion_matrix_normalized"], tmp_path / "m", normalized=True)

    assert (tmp_path / "m.png").is_file()
    assert (tmp_path / "m.pdf").is_file()