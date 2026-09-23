"""Trains a CatBoost candidate, validates it with a temporal split (never a
random split -- see guia metodologica p.6), compares it against the
shift-24h baseline on the same validation slice using the official metric,
and only promotes it to champion if it both beats the baseline AND
completes a dry-run inference. Novelty alone is never grounds to promote
(guia metodologica p.11).
"""
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


def official_accuracy(y_true: pd.Series, y_pred: pd.Series, station_ids: pd.Series) -> float:
    """Mean of per-station accuracy = 100 * max(0, 1 - WAPE), unweighted
    across stations -- exactly the competition metric (guia p.12)."""
    per_station = []
    for sid in sorted(station_ids.unique()):
        mask = station_ids == sid
        real = y_true[mask]
        pred = y_pred[mask]
        denom = real.abs().sum()
        if denom == 0:
            continue
        wape = (real - pred).abs().sum() / denom
        per_station.append(100 * max(0.0, 1 - wape))
    return float(np.mean(per_station)) if per_station else 0.0


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def run() -> int:
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

        cat_idx = [features.FEATURE_COLUMNS.index(c) for c in features.CATEGORICAL_COLUMNS]
        model = CatBoostRegressor(
            iterations=400,
            depth=6,
            learning_rate=0.08,
            loss_function="MAE",
            random_seed=20260916,
            verbose=False,
        )
        model.fit(Pool(X_train, y_train, cat_features=cat_idx))

        candidate_pred = pd.Series(np.clip(model.predict(X_valid), 0, None), index=valid_df.index)
        candidate_accuracy = official_accuracy(y_valid, candidate_pred, valid_df["station_id"])

        baseline_pred = valid_df["lag_1440"]  # shift-24h baseline, same feature already computed
        baseline_accuracy = official_accuracy(y_valid, baseline_pred, valid_df["station_id"])

        print(f"train: validation accuracy -- candidate={candidate_accuracy:.2f}  baseline={baseline_accuracy:.2f}")

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
            "baseline_accuracy": baseline_accuracy,
            "validation_days": VALIDATION_DAYS,
            "validation_rows": len(valid_df),
            "train_rows": len(train_df),
            "dry_run_ok": dry_run_ok,
            "git_commit": commit,
        }
        feature_set = {"columns": features.FEATURE_COLUMNS, "categorical": features.CATEGORICAL_COLUMNS}

        promote = candidate_accuracy > baseline_accuracy and dry_run_ok

        artifact_path = f"{ARTIFACT_DIR}/{version}.cbm"
        model.save_model(artifact_path)
        print(f"train: candidate artifact saved to {artifact_path}")

        if promote:
            db.promote_model(conn, version, trained_at, feature_set, metrics_summary)
            print(f"train: PROMOTED {version} to champion "
                  f"({candidate_accuracy:.2f} > baseline {baseline_accuracy:.2f}, dry run ok)")
        else:
            db.register_candidate(conn, version, trained_at, feature_set, metrics_summary)
            reason = "did not beat baseline" if candidate_accuracy <= baseline_accuracy else "dry-run inference failed"
            print(f"train: NOT promoted ({reason}). Champion unchanged. Candidate kept as evidence.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
