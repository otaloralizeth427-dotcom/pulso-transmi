"""MLflow experiment tracking for training runs.

Optional by design: Supabase's model_state stays the source of truth the
pipeline reads (champion, drift baseline), and MLflow is the experiment
log on top of it. If mlflow isn't installed, or MLFLOW_TRACKING_URI isn't
set, log_training_run() is a no-op -- the pipeline never fails because
tracking is unavailable.
"""
import os
from datetime import datetime

EXPERIMENT_NAME = "pulso-transmi-demand"


def _mlflow():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def log_training_run(version: str, trained_at: datetime | str, metrics_summary: dict, feature_set: dict,
                     promoted: bool, extra_tags: dict | None = None) -> bool:
    mlflow = _mlflow()
    if mlflow is None:
        return False

    if isinstance(trained_at, str):
        trained_at = datetime.fromisoformat(trained_at)
    ms = metrics_summary or {}
    drift_stations = ms.get("drift_stations") or []
    tags = {
        "model_version": version,
        "model_family": "catboost" if version.startswith("catboost-") else "baseline",
        "promoted": str(promoted).lower(),
        "trigger": "drift" if drift_stations else "scheduled/manual",
        "git_commit": ms.get("git_commit") or "",
        "champion_version_compared": ms.get("champion_version_compared") or "",
        **(extra_tags or {}),
    }
    params = {
        "validation_days": ms.get("validation_days"),
        "train_rows": ms.get("train_rows"),
        "validation_rows": ms.get("validation_rows"),
        "training_data_end": ms.get("training_data_end"),
        "n_features": len((feature_set or {}).get("columns") or []),
        "drift_stations": ",".join(drift_stations) if drift_stations else "none",
    }
    metrics = {
        "accuracy_candidate": ms.get("candidate_accuracy"),
        "accuracy_baseline_shift24h": ms.get("baseline_accuracy"),
        "accuracy_baseline_rollmean1h": ms.get("baseline2_accuracy"),
        "accuracy_champion": ms.get("champion_accuracy"),
    }
    if ms.get("candidate_accuracy") is not None and ms.get("baseline_accuracy") is not None:
        metrics["lift_vs_baseline"] = ms["candidate_accuracy"] - ms["baseline_accuracy"]
    for station, acc in (ms.get("candidate_per_station_accuracy") or {}).items():
        metrics[f"accuracy_station_{station}"] = acc

    client = mlflow.MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    run = client.create_run(experiment.experiment_id, start_time=int(trained_at.timestamp() * 1000),
                            tags={k: str(v) for k, v in tags.items()}, run_name=version)
    run_id = run.info.run_id
    for k, v in params.items():
        if v is not None:
            client.log_param(run_id, k, v)
    for k, v in metrics.items():
        if v is not None:
            client.log_metric(run_id, k, float(v))
    client.set_terminated(run_id, end_time=int(trained_at.timestamp() * 1000))
    return True
