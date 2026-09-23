import pandas as pd

import features


def _make_series(start: str, periods: int, values) -> pd.Series:
    idx = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    return pd.Series(values, index=idx)


def test_build_feature_row_none_without_enough_lookback():
    series = _make_series("2026-01-01T00:00:00Z", 10, range(10))
    origin_at = series.index[5]  # only ~1h15 of lookback, not 24h
    assert features.build_feature_row(series, None, origin_at) is None


def test_build_feature_row_has_expected_keys_with_full_history():
    periods = 24 * 4 + 10  # a bit over 24h of 15-min steps
    series = _make_series("2026-01-01T00:00:00Z", periods, range(periods))
    origin_at = series.index[-1]
    row = features.build_feature_row(series, None, origin_at)
    assert row is not None
    for mins in features.LAG_STEPS_MINUTES:
        assert f"lag_{mins}" in row
    assert row["lag_0"] == series.loc[origin_at]
    lag_24h = origin_at - pd.Timedelta(hours=24)
    assert row["lag_1440"] == series.loc[lag_24h]
