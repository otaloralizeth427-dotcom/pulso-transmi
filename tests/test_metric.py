import pandas as pd

from train import official_accuracy


def test_official_accuracy_perfect_prediction_is_100():
    y = pd.Series([10.0, 20.0, 30.0])
    pred = pd.Series([10.0, 20.0, 30.0])
    stations = pd.Series(["A", "A", "B"])
    assert official_accuracy(y, pred, stations) == 100.0


def test_official_accuracy_unweighted_across_stations():
    # station A: perfect. station B: 100% off. Unweighted mean must be 50,
    # never skewed toward whichever station has more rows/volume
    # (guia metodologica p.12).
    y = pd.Series([10.0, 10.0, 10.0, 100.0])
    pred = pd.Series([10.0, 10.0, 10.0, 0.0])
    stations = pd.Series(["A", "A", "A", "B"])
    assert official_accuracy(y, pred, stations) == 50.0
