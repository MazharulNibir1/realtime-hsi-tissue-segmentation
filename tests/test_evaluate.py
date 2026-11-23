import numpy as np
import pytest

from hsi.evaluate import aggregate, top_confusions


def _run(model, bands, seed, f1, cm):
    return {"model": model, "bands": bands, "seed": seed, "macro_f1": f1, "accuracy": f1,
            "image_accuracy": 1.0, "per_class_f1": [f1, f1], "confusion_matrix": cm,
            "per_pig_macro_f1": {"P1": f1}}  # fmt: skip


def test_aggregate_mean_std_and_pooled_cm():
    cm = [[8, 2], [0, 10]]
    agg = aggregate([_run("m", "full", 0, 0.8, cm), _run("m", "full", 1, 0.9, cm)])
    a = agg["m/full"]
    assert a["n_seeds"] == 2
    assert a["macro_f1"]["mean"] == pytest.approx(0.85)
    assert a["macro_f1"]["std"] == pytest.approx(np.std([0.8, 0.9], ddof=1))
    assert a["confusion_matrix_sum"] == [[16, 4], [0, 20]]


def test_top_confusions_row_normalised():
    cm = np.array([[6, 4, 0], [0, 10, 0], [1, 0, 9]])
    (t, p, v), *_ = top_confusions(cm, ["a", "b", "c"], 2)
    assert (t, p) == ("a", "b") and v == pytest.approx(0.4)
