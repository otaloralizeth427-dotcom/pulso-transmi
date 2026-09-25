"""Inference + submission. Consults the live cycle for data_cutoff, targets
and deadline -- never derives them locally (guia metodologica p.9) -- loads
whichever model is currently the champion, builds the exact 48 requested
predictions, and submits with a stable idempotency key so a retry never
creates a duplicate entry (guia p.9, p.13).
"""
import sys
from datetime import datetime, timezone

import pandas as pd

import db
import features
from api_client import PulsoTransmiClient
from config import PTM_API_KEY
from train import git_commit


def baseline_predict(series: pd.Series, target_at: pd.Timestamp) -> float | None:
    lag_at = target_at - pd.Timedelta(hours=24)
    if lag_at not in series.index:
        return None
    return float(series.loc[lag_at])


NO_OPEN_CYCLE = 2  # distinct from success (0) and failure (1) -- callers that
                    # poll for an open cycle need to tell "nothing to do yet"
                    # apart from "actually submitted".


def run() -> int:
    if not PTM_API_KEY:
        print("predict: PTM_API_KEY is not set, cannot submit. Aborting.")
        return 1

    client = PulsoTransmiClient()
    cycle = client.current_cycle()
    if cycle is None or cycle.get("state") != "open":
        print(f"predict: no open cycle right now (state={cycle.get('state') if cycle else None}); nothing to do.")
        return NO_OPEN_CYCLE

    cycle_id = cycle["cycle_id"]
    data_cutoff = cycle["data_cutoff"]
    targets = cycle["targets"]
    origin_at = pd.Timestamp(cycle["origin_at"])

    with db.connect() as conn:
        run_id = db.start_pipeline_run(conn, data_cutoff, f"predict for {cycle_id}")

        active = db.get_active_model(conn) if conn is not None else None
        model_version = active["model_version"] if active else "baseline-shift24h-v2"
        is_catboost = model_version.startswith("catboost-")

        obs_df = features.load_observations_df(conn) if conn is not None else None
        if obs_df is None or obs_df.empty:
            print("predict: no observations available locally (SUPABASE_DB_URL unset or empty table); "
                  "cannot build any prediction. Run ingest.py first.")
            db.finish_pipeline_run(conn, run_id, "failed", "no observations available")
            return 1

        context_df = features.load_context_df(conn)
        context_series = context_df.set_index(pd.to_datetime(context_df["observed_at"], utc=True))

        series_by_station = {}
        for sid in sorted(obs_df.station_id.unique()):
            s = obs_df[obs_df.station_id == sid].set_index("observed_at")["demand"]
            s.index = pd.to_datetime(s.index, utc=True)
            series_by_station[sid] = s.sort_index()

        model = None
        if is_catboost:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(f"artifacts/models/{model_version}.cbm")

        predictions = []
        prediction_log_rows = []
        skipped = 0
        for t in targets:
            sid = t["station_id"]
            target_at = pd.Timestamp(t["target_at"])
            series = series_by_station.get(sid)
            if series is None:
                skipped += 1
                continue

            if is_catboost:
                row = features.build_inference_row(series, context_series, sid, origin_at, t["horizon_minutes"])
                if row is None:
                    value = baseline_predict(series, target_at)  # fall back per-target rather than fail the batch
                else:
                    X = pd.DataFrame([row])[features.FEATURE_COLUMNS]
                    value = float(model.predict(X)[0])
            else:
                value = baseline_predict(series, target_at)

            if value is None:
                skipped += 1
                continue

            value = max(0.0, float(value))
            predictions.append({"station_id": sid, "target_at": t["target_at"], "value": value})
            prediction_log_rows.append({
                "station_id": sid,
                "horizon_minutes": t["horizon_minutes"],
                "predicted_for": t["target_at"],
                "predicted_demand": value,
            })

        expected = len(targets)
        if len(predictions) != expected:
            msg = f"predict: only built {len(predictions)}/{expected} predictions (skipped {skipped}); refusing to submit a partial batch."
            print(msg)
            db.finish_pipeline_run(conn, run_id, "failed", msg)
            return 1

        payload = {
            "schema_version": "1.0",
            "cycle_id": cycle_id,
            "client_run_id": f"{model_version}-{cycle_id}"[:128],
            "data_cutoff": data_cutoff,
            "model": {
                "version": model_version,
                "trained_at": (active["trained_at"].isoformat() if active and active.get("trained_at") else None),
                "training_data_end": data_cutoff,
                "git_commit": git_commit(),
            },
            "predictions": predictions,
        }

        idem_key = f"auto-{model_version}-{cycle_id}"[:128]
        status, resp = client.submit(payload, idem_key)
        print(f"predict: submit status={status} response={resp}")

        # 201 = new submission accepted. 200 = idempotent replay of a submission
        # this same run already made (same Idempotency-Key, same payload) --
        # also a success, not a failure (guia p.9: "repetir con la misma llave
        # devuelve el mismo recibo, sin duplicarla").
        if status in (200, 201):
            db.log_predictions(conn, run_id, prediction_log_rows)
            db.finish_pipeline_run(conn, run_id, "success",
                                    f"submission_id={resp.get('submission_id')} model={model_version} http={status}")
        else:
            db.finish_pipeline_run(conn, run_id, "failed", f"submit rejected: {resp}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(run())
