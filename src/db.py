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
            "select model_version, trained_at, feature_set, metrics_summary "
            "from model_state where is_active = true order by trained_at desc limit 1"
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"model_version": row[0], "trained_at": row[1], "feature_set": row[2], "metrics_summary": row[3]}


def promote_model(conn, model_version: str, trained_at: str, feature_set: dict, metrics_summary: dict) -> None:
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute("update model_state set is_active = false where is_active = true")
        cur.execute(
            """
            insert into model_state (model_version, trained_at, is_active, feature_set, metrics_summary)
            values (%s, %s, true, %s, %s)
            """,
            (model_version, trained_at, psycopg2_json(feature_set), psycopg2_json(metrics_summary)),
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


def psycopg2_json(value: dict):
    import json
    from psycopg2.extras import Json

    return Json(json.loads(json.dumps(value, default=str)))
