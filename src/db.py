"""Thin optional Postgres helper.

Every function here degrades gracefully when SUPABASE_DB_URL isn't set: the
pipeline must keep submitting predictions to the competition even if local
traceability logging isn't configured yet.
"""
import contextlib

from config import SUPABASE_DB_URL


def available() -> bool:
    return bool(SUPABASE_DB_URL)


@contextlib.contextmanager
def connect():
    if not SUPABASE_DB_URL:
        yield None
        return
    import psycopg2

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_ingestion_cursor(conn, source: str) -> str | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute("select cursor from ingestion_state where source = %s", (source,))
        row = cur.fetchone()
        return row[0] if row else None


def set_ingestion_cursor(conn, source: str, cursor: str, last_observed_at: str | None) -> None:
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into ingestion_state (source, cursor, last_observed_at, updated_at)
            values (%s, %s, %s, now())
            on conflict (source) do update
                set cursor = excluded.cursor,
                    last_observed_at = excluded.last_observed_at,
                    updated_at = now()
            """,
            (source, cursor, last_observed_at),
        )


def upsert_observations(conn, rows: list[dict]) -> None:
    if conn is None or not rows:
        return
    from psycopg2.extras import execute_values

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            insert into observations (station_id, observed_at, demand)
            values %s
            on conflict (station_id, observed_at) do update set demand = excluded.demand
            """,
            [(r["station_id"], r["observed_at"], r["demand"]) for r in rows],
            page_size=1000,
        )


def already_submitted(conn, cycle_id: str) -> bool:
    """True if THIS cycle already has a successful submission on record --
    checked against Supabase, not in-process memory. watch_and_submit.py's
    self-retriggering means a brand new job (fresh process, no memory of
    what a PREVIOUS job already submitted) can pick up a cycle that's still
    technically open per the API and try to resubmit it with slightly
    different data, tripping the API's idempotency_conflict check on every
    retry. Caught live: a cycle submitted successfully at 22:48 got
    retried 8 times starting at 23:08 by a new job that had no way to know
    it was already handled."""
    if conn is None:
        return False
    with conn.cursor() as cur:
        cur.execute(
            "select 1 from pipeline_runs where status = 'success' and notes like %s limit 1",
            (f"predict for {cycle_id} | submission_id=%",),
        )
        return cur.fetchone() is not None


def start_pipeline_run(conn, data_cutoff: str | None, notes: str) -> str | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into pipeline_runs (started_at, status, data_cutoff, notes)
            values (now(), 'running', %s, %s)
            returning id
            """,
            (data_cutoff, notes),
        )
        return str(cur.fetchone()[0])


def finish_pipeline_run(conn, run_id: str | None, status: str, notes: str) -> None:
    if conn is None or run_id is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            "update pipeline_runs set finished_at = now(), status = %s, notes = notes || ' | ' || %s where id = %s",
            (status, notes, run_id),
        )


def log_predictions(conn, run_id: str | None, rows: list[dict]) -> None:
    if conn is None or run_id is None or not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into predictions (run_id, station_id, horizon_minutes, predicted_for, predicted_demand)
            values (%(run_id)s, %(station_id)s, %(horizon_minutes)s, %(predicted_for)s, %(predicted_demand)s)
            """,
            [{**r, "run_id": run_id} for r in rows],
        )


def get_active_model(conn) -> dict | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "select model_version, trained_at, feature_set, metrics_summary, psi_reference "
            "from model_state where is_active = true order by trained_at desc limit 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"model_version": row[0], "trained_at": row[1], "feature_set": row[2], "metrics_summary": row[3],
                "psi_reference": row[4]}


def promote_model(conn, model_version: str, trained_at: str, feature_set: dict, metrics_summary: dict,
                   psi_reference: dict | None = None) -> None:
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute("update model_state set is_active = false where is_active = true")
        cur.execute(
            """
            insert into model_state (model_version, trained_at, is_active, feature_set, metrics_summary, psi_reference)
            values (%s, %s, true, %s, %s, %s)
            """,
            (model_version, trained_at, psycopg2_json(feature_set), psycopg2_json(metrics_summary),
             psycopg2_json(psi_reference) if psi_reference is not None else None),
        )


def register_candidate(conn, model_version: str, trained_at: str, feature_set: dict, metrics_summary: dict) -> None:
    """Store a trained-but-not-promoted candidate as evidence (is_active stays false)."""
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into model_state (model_version, trained_at, is_active, feature_set, metrics_summary)
            values (%s, %s, false, %s, %s)
            """,
            (model_version, trained_at, psycopg2_json(feature_set), psycopg2_json(metrics_summary)),
        )


def log_validation_metrics(conn, run_id: str | None, rows: list[dict]) -> None:
    """`run_id` is only used to fill rows that don't already carry their own
    (each row may belong to a different pipeline run, e.g. when evaluating
    several resolved cycles in one monitor pass)."""
    if conn is None or not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            insert into validation_metrics (run_id, station_id, horizon_minutes, wape, accuracy, computed_at)
            values (%(run_id)s, %(station_id)s, %(horizon_minutes)s, %(wape)s, %(accuracy)s, now())
            """,
            [{"run_id": run_id, **r} for r in rows],
        )


def log_leaderboard_snapshot(conn, window_kind: str, accuracy, coverage, rank, signal: str) -> None:
    """Feeds the read-only dashboard's leaderboard-position widget. Written
    with the same service-role DB connection everything else uses, so the
    submissions API key never has to reach the browser to show this."""
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into leaderboard_snapshots (window_kind, accuracy, coverage, rank, signal, checked_at)
            values (%s, %s, %s, %s, %s, now())
            """,
            (window_kind, accuracy, coverage, rank, signal),
        )


def log_drift_event(conn, *, run_id, cycle_id, station_id, champion_version, psi_flag, psi_max_feature,
                     psi_max_value, psi_details, performance_flag, performance_confirmed, rolling_wape_24h,
                     champion_valid_accuracy, n_cycles_confirming) -> str | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into drift_events (run_id, cycle_id, station_id, champion_version, psi_flag, psi_max_feature,
                psi_max_value, psi_details, performance_flag, performance_confirmed, rolling_wape_24h,
                champion_valid_accuracy, n_cycles_confirming)
            values (%(run_id)s, %(cycle_id)s, %(station_id)s, %(champion_version)s, %(psi_flag)s,
                %(psi_max_feature)s, %(psi_max_value)s, %(psi_details)s, %(performance_flag)s,
                %(performance_confirmed)s, %(rolling_wape_24h)s, %(champion_valid_accuracy)s,
                %(n_cycles_confirming)s)
            returning id
            """,
            {
                "run_id": run_id, "cycle_id": cycle_id, "station_id": station_id,
                "champion_version": champion_version, "psi_flag": psi_flag, "psi_max_feature": psi_max_feature,
                "psi_max_value": psi_max_value, "psi_details": psycopg2_json(psi_details) if psi_details else None,
                "performance_flag": performance_flag, "performance_confirmed": performance_confirmed,
                "rolling_wape_24h": rolling_wape_24h, "champion_valid_accuracy": champion_valid_accuracy,
                "n_cycles_confirming": n_cycles_confirming,
            },
        )
        return str(cur.fetchone()[0])


def get_recent_performance_flags(conn, station_id: str, n: int) -> list[bool]:
    """Last `n` performance_flag values for this station, most recent first
    -- used to confirm a drift streak instead of reacting to one noisy cycle."""
    if conn is None:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "select performance_flag from drift_events where station_id = %s order by checked_at desc limit %s",
            (station_id, n),
        )
        return [row[0] for row in cur.fetchall()]


def get_last_retrain_trigger_time(conn):
    """Anchors the cooldown guard on DISPATCH time (triggered_at), not on
    when a past retrain finished or promoted -- a long-running retrain
    can't get overlapped by a second trigger fired while it's still going."""
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute("select triggered_at from retrain_triggers order by triggered_at desc limit 1")
        row = cur.fetchone()
        return row[0] if row else None


def insert_retrain_trigger(conn, drift_event_id: str, stations: list[str], reason: str) -> str | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into retrain_triggers (triggered_at, drift_event_id, stations, reason, status)
            values (now(), %s, %s, %s, 'dispatched')
            returning id
            """,
            (drift_event_id, stations, reason),
        )
        return str(cur.fetchone()[0])


def complete_retrain_trigger(conn, trigger_id: str, *, promoted: bool, candidate_version: str,
                              candidate_accuracy: float, champion_accuracy: float) -> None:
    if conn is None or not trigger_id:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            update retrain_triggers
            set status = 'completed', completed_at = now(), promoted = %s, candidate_version = %s,
                candidate_accuracy = %s, champion_accuracy = %s
            where id = %s
            """,
            (promoted, candidate_version, candidate_accuracy, champion_accuracy, trigger_id),
        )


def count_new_observations(conn, since) -> int:
    """Total new observation rows across all stations since `since` --
    the minimum-new-data guard before allowing a drift-triggered retrain."""
    if conn is None:
        return 0
    with conn.cursor() as cur:
        cur.execute("select count(*) from observations where observed_at > %s", (since,))
        return cur.fetchone()[0]


def psycopg2_json(value: dict):
    import json
    from psycopg2.extras import Json

    return Json(json.loads(json.dumps(value, default=str)))
