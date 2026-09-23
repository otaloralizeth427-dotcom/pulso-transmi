"""Evaluation + drift monitoring.

Local step: for predictions whose target has since been observed, compute a
per-station/horizon WAPE-based accuracy and log it to validation_metrics --
this is a diagnostic granularity to spot *which* station/horizon degrades,
not the official score (that's computed server-side).

Drift step: compares this participant's official rolling-24h accuracy
against their cumulative accuracy from /v1/leaderboard. The 5-point
threshold below is a starting point, not a validated number -- the guide
is explicit that drift thresholds must be justified, not copied
(guia metodologica p.17). Revisit it once there's enough history to see
what normal cycle-to-cycle noise looks like.
"""
import sys

import db
from api_client import PulsoTransmiClient

DRIFT_THRESHOLD_POINTS = 5.0


def evaluate_resolved_predictions(conn) -> int:
    if conn is None:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            select p.run_id, p.station_id, p.horizon_minutes, p.predicted_demand, o.demand
            from predictions p
            join observations o
              on o.station_id = p.station_id and o.observed_at = p.predicted_for
            where not exists (
                select 1 from validation_metrics vm
                where vm.run_id = p.run_id
                  and vm.station_id = p.station_id
                  and vm.horizon_minutes = p.horizon_minutes
            )
            """
        )
        rows = cur.fetchall()

    metrics = []
    for run_id, station_id, horizon_minutes, predicted, real in rows:
        denom = abs(real) if real else 0
        wape = abs(real - predicted) / denom if denom else (0.0 if predicted == 0 else 1.0)
        accuracy = 100 * max(0.0, 1 - wape)
        metrics.append({
            "run_id": run_id,
            "station_id": station_id,
            "horizon_minutes": horizon_minutes,
            "wape": wape,
            "accuracy": accuracy,
        })

    if metrics:
        db.log_validation_metrics(conn, None, metrics)  # run_id already embedded per-row below
        # log_validation_metrics ignores its run_id arg when rows already carry one;
        # keep the explicit column so each row is attributed to its own run.
    return len(metrics)


def check_drift(client: PulsoTransmiClient) -> dict:
    me = client.session.get(f"{client.base_url}/v1/me", headers=client.auth_headers(), timeout=20).json()
    participant_id = me.get("participant_id")

    cumulative = client.leaderboard(window="cumulative")
    rolling = client.leaderboard(window="rolling_24h")

    def find_score(board: dict) -> float | None:
        for entry in board.get("entries", board.get("leaderboard", [])):
            if entry.get("participant_id") == participant_id:
                return entry.get("accuracy")
        return None

    cum_score = find_score(cumulative)
    roll_score = find_score(rolling)

    signal = "not_enough_data"
    if cum_score is not None and roll_score is not None:
        drop = cum_score - roll_score
        signal = "performance_drift" if drop >= DRIFT_THRESHOLD_POINTS else "stable"
        print(f"monitor: cumulative={cum_score:.2f} rolling_24h={roll_score:.2f} drop={drop:.2f} -> {signal}")
    else:
        print(f"monitor: not enough leaderboard history yet to compare (cumulative={cum_score}, rolling={roll_score})")

    return {"cumulative_accuracy": cum_score, "rolling_24h_accuracy": roll_score, "signal": signal}


def run() -> int:
    client = PulsoTransmiClient()
    with db.connect() as conn:
        n = evaluate_resolved_predictions(conn)
        print(f"monitor: logged {n} newly-resolved validation_metrics row(s)")

    drift = check_drift(client)
    if drift["signal"] == "performance_drift":
        print("monitor: PERFORMANCE DRIFT signal raised -- consider training a new candidate "
              "(train.py will only promote it if it actually beats the current champion).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
