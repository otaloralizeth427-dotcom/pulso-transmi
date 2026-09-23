"""Feature construction shared by training and inference.

Every feature here is computable from data available strictly at-or-before
the row's `origin_at` (the cycle's data_cutoff at inference time), so the
same function can build both the training frame and a live inference row
without leaking the future.
"""
import pandas as pd

LAG_STEPS_MINUTES = [0, 15, 30, 45, 60, 24 * 60]  # last: same time yesterday
ROLL_WINDOWS_STEPS = [4, 96]  # 1h, 24h (15-min cadence)
HORIZONS_MINUTES = [15, 30, 45, 60]
STEP = pd.Timedelta(minutes=15)


def load_observations_df(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("select station_id, observed_at, demand from observations order by station_id, observed_at")
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["station_id", "observed_at", "demand"])


def load_context_df(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "select observed_at, rain_mm, temperature_c, event_intensity from context order by observed_at"
        )
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["observed_at", "rain_mm", "temperature_c", "event_intensity"])


def _station_series(obs_df: pd.DataFrame, station_id: str) -> pd.Series:
    s = obs_df[obs_df.station_id == station_id].set_index("observed_at")["demand"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index()


def _calendar_features(ts: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "hour": ts.hour,
        "minute": ts.minute,
        "dow": ts.dayofweek,
    }, index=ts)


def build_feature_row(series: pd.Series, context: pd.Series | None, origin_at) -> dict | None:
    """Build one feature row from a station's demand series, as of origin_at.

    Returns None if there isn't enough lookback (24h + buffer) to compute
    every lag feature -- callers should skip that row rather than impute it.
    """
    origin_at = pd.Timestamp(origin_at)
    feats = {}
    for mins in LAG_STEPS_MINUTES:
        ts = origin_at - pd.Timedelta(minutes=mins)
        if ts not in series.index:
            return None
        feats[f"lag_{mins}"] = series.loc[ts]

    for window in ROLL_WINDOWS_STEPS:
        window_ts = [origin_at - i * STEP for i in range(window)]
        vals = series.reindex(window_ts)
        if vals.isna().any():
            return None
        feats[f"roll_mean_{window}"] = vals.mean()

    feats["hour"] = origin_at.hour
    feats["minute"] = origin_at.minute
    feats["dow"] = origin_at.dayofweek

    if context is not None and origin_at in context.index:
        row = context.loc[origin_at]
        feats["rain_mm"] = row.get("rain_mm", 0.0) or 0.0
        feats["temperature_c"] = row.get("temperature_c", 0.0) or 0.0
        feats["event_intensity"] = row.get("event_intensity", 0.0) or 0.0
    else:
        feats["rain_mm"] = 0.0
        feats["temperature_c"] = 0.0
        feats["event_intensity"] = 0.0

    return feats


FEATURE_COLUMNS = (
    [f"lag_{m}" for m in LAG_STEPS_MINUTES]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS_STEPS]
    + ["hour", "minute", "dow", "rain_mm", "temperature_c", "event_intensity", "station_id", "horizon_minutes"]
)
CATEGORICAL_COLUMNS = ["station_id", "horizon_minutes"]


def build_training_frame(obs_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    context_series = context_df.set_index(pd.to_datetime(context_df["observed_at"], utc=True))
    station_ids = sorted(obs_df.station_id.unique())

    examples = []
    for sid in station_ids:
        series = _station_series(obs_df, sid)
        # candidate origins: every timestamp that has a value, skipping the
        # first 24h (not enough lookback) and last 1h (not enough horizon).
        candidate_origins = series.index[(series.index >= series.index.min() + pd.Timedelta(hours=25))
                                          & (series.index <= series.index.max() - pd.Timedelta(hours=1))]
        for origin_at in candidate_origins:
            feats = build_feature_row(series, context_series, origin_at)
            if feats is None:
                continue
            for h in HORIZONS_MINUTES:
                target_at = origin_at + pd.Timedelta(minutes=h)
                if target_at not in series.index:
                    continue
                row = dict(feats)
                row["station_id"] = sid
                row["horizon_minutes"] = h
                row["origin_at"] = origin_at
                row["target_at"] = target_at
                row["y"] = series.loc[target_at]
                examples.append(row)
    return pd.DataFrame(examples)


def build_inference_row(series: pd.Series, context_series: pd.DataFrame | None, station_id: str,
                         origin_at, horizon_minutes: int) -> dict | None:
    feats = build_feature_row(series, context_series, origin_at)
    if feats is None:
        return None
    feats["station_id"] = station_id
    feats["horizon_minutes"] = horizon_minutes
    return feats
