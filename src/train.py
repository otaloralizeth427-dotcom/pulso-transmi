"""Trains a CatBoost candidate, validates it with a temporal split (never a
random split -- see guia metodologica p.6), and only promotes it to
champion if it beats BOTH the shift-24h baseline and the currently active
champion on the same validation slice, plus completes a dry-run inference.
Novelty alone is never grounds to promote (guia metodologica p.11).

Earlier versions of this script only required beating the naive baseline,
not the live champion -- since CatBoost reliably clears that low bar, every
retraining run promoted automatically regardless of whether the new
candidate was actually better than what was already deployed. Caught this
the first time a retrain produced a candidate (81.71) slightly below the
previous champion's own validation score (82.46) and it still promoted.

Two things only matter for drift-triggered retrains (src/drift.py), both
opt-in via environment variables so the plain weekly/manual run is
unaffected:

- DRIFT_STATIONS: comma-separated station_ids to upweight in training, so a
  retrain triggered by drift in 2 stations doesn't get diluted by treating
  all 12 equally.
- RETRAIN_TRIGGER_ID: the retrain_triggers.id this run is fulfilling, so its
  outcome (promoted or not, candidate/champion accuracy) gets written back
  for the dashboard -- the cooldown guard in drift.py already anchors on
  when that row's trigger was DISPATCHED, not on this completing.

On promotion, also stores per-station validation accuracy and a fresh PSI
reference distribution (computed from THIS training window) on the
model_state row -- drift.py always reads the reference off the currently
active champion, so promoting a new one updates the drift baseline
automatically instead of comparing against the very first model forever.
"""
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import db
import features

VALIDATION_DAYS = 7
ARTIFACT_DIR = "artifacts/models"
PSI_FEATURES = ["ratio_vs_yesterday"]  # deseasonalized signal: lag_0 /
# lag_1440 (now vs the same time 24h earlier). Raw demand/roll_mean_96
# LEVELS carry strong day/night structure, and a short "current" window
# (drift.py looks back a few days) will never reproduce a multi-week
# reference's day/night mixture -- that alone produced PSI > 1 on every
# single station with zero real drift injected (caught testing against
# live data before shipping this). Dividing by the same time yesterday
# cancels daily seasonality, so what's left reflects an actual level shift,
# not window sampling. Station-specific only -- context (rain_mm/
# temperature_c/event_intensity) has no station_id in this schema, so a
# "per-station" PSI on it would read identical across all 12 stations.
PSI_BINS = 10
DRIFT_WEIGHT_MULTIPLIER = 3.0
DRIFT_RECENCY_HALF_LIFE_DAYS = 7  # exponential recency weight for a
# drift-triggered retrain: weight = 0.5 ** (age_days / half_life). Tried a
# hard window cutoff first (train only on the last N days) -- it still
# lost to the champion, because the real injected drift turned out to be
# only ~19 hours old at the time: even a 14-day window was 94% stale data.
# Tested half-lives from 1 to 999 days against live data once that was
# known: very aggressive decay (1d) UNDERPERFORMED a plain full-history fit
# -- with that little genuine post-drift signal, leaning on it exclusively
# starves the model of the broader day-of-week/hour structure it still
# needs. 7d gave the best validation accuracy among those tested (noisy
# signal, not a sharp optimum) -- moderate decay, not aggressive. Revisit
# once more post-drift data has actually accumulated.


def per_station_accuracy(y_true: pd.Series, y_pred: pd.Series, station_ids: pd.Series) -> dict:
    result = {}
    for sid in sorted(station_ids.unique()):
        mask = station_ids == sid
        real = y_true[mask]
        pred = y_pred[mask]
        denom = real.abs().sum()
        if denom == 0:
            continue
        wape = (real - pred).abs().sum() / denom
        result[sid] = 100 * max(0.0, 1 - wape)
    return result


def official_accuracy(y_true: pd.Series, y_pred: pd.Series, station_ids: pd.Series) -> float:
    """Mean of per-station accuracy = 100 * max(0, 1 - WAPE), unweighted
    across stations -- exactly the competition metric (guia p.12)."""
    per_station = per_station_accuracy(y_true, y_pred, station_ids)
    return float(np.mean(list(per_station.values()))) if per_station else 0.0


def compute_psi_reference(train_df: pd.DataFrame) -> dict:
    """Per-station quantile bin edges for PSI_FEATURES, from this training
    window. Open-ended outer edges so a future value outside the observed
    range still lands in the nearest bin instead of being dropped."""
    reference = {}
    eps = 1e-6
    for sid in sorted(train_df["station_id"].unique()):
        sub = train_df[train_df.station_id == sid]
        ratio = sub["lag_0"] / sub["lag_1440"].clip(lower=eps)
        station_ref = {}
        vals = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) >= PSI_BINS:
            edges = list(np.quantile(vals, np.linspace(0, 1, PSI_BINS + 1)))
            edges[0] = -1e18
            edges[-1] = 1e18
            station_ref["ratio_vs_yesterday"] = edges
        reference[sid] = station_ref
    return reference


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def run() -> int:
    drift_stations = {s.strip() for s in os.environ.get("DRIFT_STATIONS", "").split(",") if s.strip()}
    retrain_trigger_id = os.environ.get("RETRAIN_TRIGGER_ID")
    is_drift_retrain = bool(drift_stations)

    with db.connect() as conn:
        if conn is None:
            print("train: SUPABASE_DB_URL not set, cannot load training data. Aborting.")
            return 1

        obs_df = features.load_observations_df(conn)
        context_df = features.load_context_df(conn)
        frame = features.build_training_frame(obs_df, context_df)

        if frame.empty:
            print("train: not enough history to build any training example yet.")
            return 1

        cutoff = frame["origin_at"].max() - pd.Timedelta(days=VALIDATION_DAYS)
        train_df = frame[frame["origin_at"] < cutoff]
        valid_df = frame[frame["origin_at"] >= cutoff]

        print(f"train: {len(train_df)} training rows, {len(valid_df)} validation rows "
              f"(split at {cutoff}, last {VALIDATION_DAYS}d held out)")

        if train_df.empty or valid_df.empty:
            print("train: temporal split produced an empty side, need more history first.")
            return 1

        X_train = train_df[features.FEATURE_COLUMNS]
        y_train = train_df["y"]
        X_valid = valid_df[features.FEATURE_COLUMNS]
        y_valid = valid_df["y"]

        sample_weight = None
        if is_drift_retrain:
            age_days = (train_df["origin_at"].max() - train_df["origin_at"]).dt.total_seconds() / 86400
            sample_weight = np.exp(-np.log(2) / DRIFT_RECENCY_HALF_LIFE_DAYS * age_days).to_numpy()
            print(f"train: drift retrain -- applying recency weight (half_life={DRIFT_RECENCY_HALF_LIFE_DAYS}d), "
                  f"min={sample_weight.min():.4f} max={sample_weight.max():.4f}")
            if drift_stations:
                station_multiplier = np.where(train_df["station_id"].isin(drift_stations), DRIFT_WEIGHT_MULTIPLIER, 1.0)
                sample_weight = sample_weight * station_multiplier
                print(f"train: additionally upweighting {sorted(drift_stations)} by {DRIFT_WEIGHT_MULTIPLIER}x "
                      f"({int((station_multiplier > 1).sum())} of {len(station_multiplier)} rows)")

        cat_idx = [features.FEATURE_COLUMNS.index(c) for c in features.CATEGORICAL_COLUMNS]
        model = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.08,
            loss_function="MAE",
            random_seed=20260916,
            verbose=False,
        )
        model.fit(Pool(X_train, y_train, cat_features=cat_idx, weight=sample_weight))

        candidate_pred = pd.Series(np.clip(model.predict(X_valid), 0, None), index=valid_df.index)
        candidate_accuracy = official_accuracy(y_valid, candidate_pred, valid_df["station_id"])
        candidate_per_station = per_station_accuracy(y_valid, candidate_pred, valid_df["station_id"])

        baseline_pred = valid_df["lag_1440"]  # shift-24h baseline, same feature already computed
        baseline_accuracy = official_accuracy(y_valid, baseline_pred, valid_df["station_id"])

        active = db.get_active_model(conn)
        champion_version = active["model_version"] if active else None
        if champion_version and champion_version.startswith("catboost-"):
            champion_model = CatBoostRegressor()
            champion_model.load_model(f"{ARTIFACT_DIR}/{champion_version}.cbm")
            champion_pred = pd.Series(np.clip(champion_model.predict(X_valid), 0, None), index=valid_df.index)
            champion_accuracy = official_accuracy(y_valid, champion_pred, valid_df["station_id"])
        else:
            # No CatBoost champion deployed yet (still on the baseline) --
            # beating the baseline itself is the only bar to clear.
            champion_accuracy = baseline_accuracy

        print(f"train: validation accuracy -- candidate={candidate_accuracy:.2f}  "
              f"baseline={baseline_accuracy:.2f}  current_champion({champion_version})={champion_accuracy:.2f}")

        trained_at = datetime.now(timezone.utc).isoformat()
        commit = git_commit()
        version = f"catboost-{trained_at[:19].replace(':', '').replace('-', '')}"

        # dry-run inference: does the model produce finite, non-negative
        # predictions for a real feature row, end to end?
        dry_run_ok = False
        try:
            sample = X_valid.iloc[[0]]
            pred = model.predict(sample)
            dry_run_ok = bool(np.isfinite(pred).all() and (pred >= -1e-6).all())
        except Exception as exc:
            print(f"train: dry-run inference failed: {exc}")

        metrics_summary = {
            "candidate_accuracy": candidate_accuracy,
            "candidate_per_station_accuracy": candidate_per_station,
            "baseline_accuracy": baseline_accuracy,
            "champion_accuracy": champion_accuracy,
            "champion_version_compared": champion_version,
            # The most recent data point this model has ever seen, on the
            # competition's VIRTUAL clock -- drift.py's minimum-new-data
            # guard compares against this, never against `trained_at`
            # (wall-clock time), which lives on a completely different
            # timeline from observations.observed_at and would always read
            # as "0 new observations" forever.
            "training_data_end": frame["origin_at"].max().isoformat(),
            "validation_days": VALIDATION_DAYS,
            "validation_rows": len(valid_df),
            "train_rows": len(train_df),
            "dry_run_ok": dry_run_ok,
            "git_commit": commit,
            "drift_stations": sorted(drift_stations) if drift_stations else None,
            "retrain_trigger_id": retrain_trigger_id,
        }
        feature_set = {"columns": features.FEATURE_COLUMNS, "categorical": features.CATEGORICAL_COLUMNS}

        beats_bar = candidate_accuracy > baseline_accuracy and candidate_accuracy > champion_accuracy
        promote = beats_bar and dry_run_ok

        artifact_path = f"{ARTIFACT_DIR}/{version}.cbm"
        model.save_model(artifact_path)
        print(f"train: candidate artifact saved to {artifact_path}")

        if promote:
            psi_reference = compute_psi_reference(train_df)
            db.promote_model(conn, version, trained_at, feature_set, metrics_summary, psi_reference)
            print(f"train: PROMOTED {version} to champion "
                  f"({candidate_accuracy:.2f} > baseline {baseline_accuracy:.2f} and > "
                  f"current champion {champion_accuracy:.2f}, dry run ok)")
        else:
            db.register_candidate(conn, version, trained_at, feature_set, metrics_summary)
            if not dry_run_ok:
                reason = "dry-run inference failed"
            elif candidate_accuracy <= champion_accuracy:
                reason = f"did not beat current champion ({candidate_accuracy:.2f} <= {champion_accuracy:.2f})"
            else:
                reason = "did not beat baseline"
            print(f"train: NOT promoted ({reason}). Champion unchanged. Candidate kept as evidence.")

        if retrain_trigger_id:
            db.complete_retrain_trigger(conn, retrain_trigger_id, promoted=promote, candidate_version=version,
                                         candidate_accuracy=candidate_accuracy, champion_accuracy=champion_accuracy)

    return 0


if __name__ == "__main__":
    sys.exit(run())
