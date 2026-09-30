"""Evaluation + drift monitoring, run as the last part of the hourly
pipeline job (src/watch_and_submit.py) -- no separate scheduled workflow.

Local step: for predictions whose target has since been observed, compute a
per-station/horizon WAPE-based accuracy and log it to validation_metrics --
this is a diagnostic granularity to spot *which* station/horizon degrades,
not the official score (that's computed server-side).

Leaderboard step: records this participant's official cumulative/rolling-24h
standing (src/drift.py handles the real per-station drift detection now --
see that module for PSI + rolling-WAPE-vs-champion, confirmation streaks,
and the guarded retrain trigger).
"""
import sys

import db
import drift
from api_client import PulsoTransmiClient


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


def record_leaderboard_position(client: PulsoTransmiClient, conn) -> None:
    me = client.session.get(f"{client.base_url}/v1/me", headers=client.auth_headers(), timeout=20).json()
    display_name = me.get("display_name")

    cumulative = client.leaderboard(window="cumulative")
    rolling = client.leaderboard(window="rolling_24h")
    cum_entry = find_entry(cumulative, display_name)
    roll_entry = find_entry(rolling, display_name)

    # Persisted so the read-only dashboard can show leaderboard position
    # without ever holding the submissions API key in the browser. This is
    # informational only now -- src/drift.py owns the real drift signal.
    if cum_entry:
        db.log_leaderboard_snapshot(conn, "cumulative", cum_entry.get("accuracy"), cum_entry.get("coverage"),
                                     cum_entry.get("rank"), "n/a")
        print(f"monitor: cumulative accuracy={cum_entry.get('accuracy'):.2f} rank={cum_entry.get('rank')}")
    if roll_entry:
        db.log_leaderboard_snapshot(conn, "rolling_24h", roll_entry.get("accuracy"), roll_entry.get("coverage"),
                                     roll_entry.get("rank"), "n/a")
        print(f"monitor: rolling_24h accuracy={roll_entry.get('accuracy'):.2f} rank={roll_entry.get('rank')}")


def run() -> int:
    client = PulsoTransmiClient()
    cycle = client.current_cycle()
    cycle_id = cycle.get("cycle_id") if cycle else None

    # Three independent steps, each its OWN connection/transaction, each
    # wrapped so a failure in one can never roll back or block the others.
    # Not hypothetical -- both failure modes below actually happened on the
    # same day: a bad query in drift.py raised mid-transaction and rolled
    # back everything sharing that transaction with it (validation_metrics
    # AND the leaderboard snapshot), and separately the course's leaderboard
    # API returned a transient 500 that would have done the same if it had
    # still shared a connection with the evaluation step. The dashboard sat
    # on stale data for over a day before this was caught.
    try:
        with db.connect() as conn:
            n = evaluate_resolved_predictions(conn)
            print(f"monitor: logged {n} newly-resolved validation_metrics row(s)")
    except Exception as exc:
        print(f"monitor: evaluating resolved predictions failed, leaving it for next cycle: {exc}")

    try:
        with db.connect() as conn:
            record_leaderboard_position(client, conn)
    except Exception as exc:
        print(f"monitor: recording leaderboard position failed (likely a transient API error), "
              f"leaving it for next cycle: {exc}")

    try:
        with db.connect() as conn:
            drift.run(conn, run_id=None, cycle_id=cycle_id)
    except Exception as exc:
        print(f"monitor: drift check failed, leaving it for next cycle: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
