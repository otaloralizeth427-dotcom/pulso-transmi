"""Backfill MLflow with every training run already recorded in Supabase.

MLflow tracking was added after the competition closed; until then every
training run was logged only to model_state (candidate vs. champion vs.
baselines, per-station accuracy, git commit). This replays those real
rows into MLflow -- nothing is retrained or recomputed -- and tags each run
`source=backfill:model_state` so it's never mistaken for a live log.

Reads the public read-only view v_model_state, so it needs no admin
credentials:

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    export SUPABASE_URL=https://<ref>.supabase.co SUPABASE_ANON_KEY=...
    python src/mlflow_backfill.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os
import sys
from datetime import datetime

import requests

import tracking


def fetch_view(name: str, order: str) -> list[dict]:
    url = f"{os.environ['SUPABASE_URL']}/rest/v1/{name}?select=*&order={order}.asc"
    key = os.environ["SUPABASE_ANON_KEY"]
    res = requests.get(url, headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=30)
    res.raise_for_status()
    return res.json()


def was_promoted(ms: dict) -> bool:
    # Same rule train.py applies: beat the shift-24h baseline AND the
    # champion it was compared against (the first runs had no champion to
    # compare yet), AND pass the dry-run inference.
    cand, base, champ = ms.get("candidate_accuracy"), ms.get("baseline_accuracy"), ms.get("champion_accuracy")
    if cand is None or base is None:
        return False
    return cand > base and (champ is None or cand > champ) and ms.get("dry_run_ok", True)


def main() -> int:
    if not tracking._mlflow():
        print("mlflow_backfill: set MLFLOW_TRACKING_URI and install mlflow first.")
        return 1

    reverted = {t["candidate_version"]: t.get("notes") or "" for t in fetch_view("v_retrain_triggers", "triggered_at")
                if t.get("candidate_version") and "revertido" in (t.get("notes") or "")}

    n = 0
    for m in fetch_view("v_model_state", "trained_at"):
        ms = m.get("metrics_summary") or {}
        if ms.get("candidate_accuracy") is None:
            continue  # baseline row: no training run behind it
        extra = {"source": "backfill:model_state"}
        if m["model_version"] in reverted:
            extra["note"] = "promovido y revertido manualmente (fallo el push del artefacto)"
        tracking.log_training_run(m["model_version"], datetime.fromisoformat(m["trained_at"]), ms,
                                  m.get("feature_set") or {}, was_promoted(ms), extra)
        n += 1
    print(f"mlflow_backfill: logged {n} training runs to experiment '{tracking.EXPERIMENT_NAME}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
