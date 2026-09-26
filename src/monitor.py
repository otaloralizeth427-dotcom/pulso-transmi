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


def find_entry(board: dict, display_name: str) -> dict | None:
    # /v1/leaderboard has no participant_id in its rows -- display_name is
    # the only identifying field it returns (verified against the live
    # response, not guessed from the OpenAPI schema, which leaves this
    # endpoint's body untyped).
    for entry in board.get("data", []):
        if entry.get("display_name") == display_name:
            return entry
    return None


def check_drift(client: PulsoTransmiClient, conn) -> dict:
    me = client.session.get(f"{client.base_url}/v1/me", headers=client.auth_headers(), timeout=20).json()
    display_name = me.get("display_name")

    cumulative = client.leaderboard(window="cumulative")
    rolling = client.leaderboard(window="rolling_24h")

    cum_entry = find_entry(cumulative, display_name)
    roll_entry = find_entry(rolling, display_name)
    cum_score = cum_entry.get("accuracy") if cum_entry else None
    roll_score = roll_entry.get("accuracy") if roll_entry else None

    signal = "not_enough_data"
    if cum_score is not None and roll_score is not None:
        drop = cum_score - roll_score
        signal = "performance_drift" if drop >= DRIFT_THRESHOLD_POINTS else "stable"
        print(f"monitor: cumulative={cum_score:.2f} rolling_24h={roll_score:.2f} drop={drop:.2f} -> {signal}")
    else:
        print(f"monitor: not enough leaderboard history yet to compare (cumulative={cum_score}, rolling={roll_score})")

    # Persisted so the read-only dashboard can show leaderboard position
    # without ever holding the submissions API key in the browser.
    if cum_entry:
        db.log_leaderboard_snapshot(conn, "cumulative", cum_entry.get("accuracy"), cum_entry.get("coverage"),
                                     cum_entry.get("rank"), signal)
    if roll_entry:
        db.log_leaderboard_snapshot(conn, "rolling_24h", roll_entry.get("accuracy"), roll_entry.get("coverage"),
                                     roll_entry.get("rank"), signal)

    return {"cumulative_accuracy": cum_score, "rolling_24h_accuracy": roll_score, "signal": signal}


def run() -> int:
    client = PulsoTransmiClient()
    with db.connect() as conn:
        n = evaluate_resolved_predictions(conn)
        print(f"monitor: logged {n} newly-resolved validation_metrics row(s)")
        drift = check_drift(client, conn)

    if drift["signal"] == "performance_drift":
        print("monitor: PERFORMANCE DRIFT signal raised -- consider training a new candidate "
              "(train.py will only promote it if it actually beats the current champion).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
