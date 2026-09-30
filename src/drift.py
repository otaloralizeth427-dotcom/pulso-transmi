"""Passive drift monitoring, run as part of the hourly pipeline job (no
separate scheduled workflow -- see .github/workflows/pipeline.yml). Two
signals, both computed per station (never aggregated across all 12):

- Data drift: PSI on a deseasonalized station-specific signal
  (ratio_vs_yesterday = demand[t] / demand[t-24h]) against the ACTIVE
  champion's own training-window reference (model_state.psi_reference,
  refreshed every promotion -- train.py::compute_psi_reference). PSI > 0.2
  flags a station. (Raw demand-level PSI was tried first and produced
  PSI > 1 on every station with zero real drift injected -- see
  drift.py::_station_psi's docstring.)
- Performance drift: rolling 24h WAPE per station vs. that same champion's
  per-station validation accuracy. A single bad cycle doesn't trigger
  anything -- CONFIRM_STREAK consecutive cycles must all flag before it
  counts as "confirmed" (guia metodologica: no reaccionar a un unico
  periodo dificil).

Confirmed performance drift, past two guards (cooldown anchored on when a
retrain was last DISPATCHED, and a minimum-new-data floor), triggers
.github/workflows/train-on-drift.yml via `gh workflow run` -- a workflow
that otherwise never runs on a schedule, so four days with no drift costs
zero extra Actions minutes.
"""
import os
import subprocess

import numpy as np
import pandas as pd

import db
import features

PSI_FLAG_THRESHOLD = 0.2
PERFORMANCE_DROP_THRESHOLD = 10.0  # percentage points below the champion's
                                    # own per-station validation accuracy
CONFIRM_STREAK = 3  # consecutive cycles required (guia p.17: "reaccionar a
                     # persistencia", not a single noisy cycle)
COOLDOWN_HOURS = 8   # mid-range of the 6-12h the user asked for
MIN_NEW_OBSERVATIONS = 96  # ~1 day of 15-min data per flagged station,
                            # before a retrain is even worth attempting
RECENT_WINDOW_DAYS = 3  # how much recent data PSI's "current" side looks at


def compute_psi(edges: list[float], current_values: pd.Series) -> float:
    """Reference bins are equal-frequency deciles by construction (see
    train.py::compute_psi_reference), so the expected share per bin is
    uniform -- no need to store reference counts separately."""
    if len(current_values) == 0 or len(edges) < 2:
        return 0.0
    n_bins = len(edges) - 1
    expected_pct = np.full(n_bins, 1.0 / n_bins)
    counts, _ = np.histogram(current_values, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    actual_pct = counts / total
    eps = 1e-4
    actual_pct = np.clip(actual_pct, eps, None)
    expected_pct = np.clip(expected_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _station_psi(series: pd.Series, station_ref: dict) -> tuple[str | None, float, dict]:
    """Returns (feature with the max PSI, that max value, all per-feature values).

    ratio_vs_yesterday = demand[t] / demand[t-24h], same deseasonalized
    signal train.py::compute_psi_reference builds the reference from --
    comparing raw demand LEVELS instead produced PSI > 1 on every station
    with zero real drift, purely from a short recent window not matching a
    multi-week reference's day/night mixture.
    """
    if "ratio_vs_yesterday" not in station_ref:
        return None, 0.0, {}

    cutoff = series.index.max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    lag_24h = series.shift(freq="24h")
    ratio = (series / lag_24h.reindex(series.index).clip(lower=1e-6)).replace([np.inf, -np.inf], np.nan)
    recent_ratio = ratio[ratio.index >= cutoff].dropna()

    details = {"ratio_vs_yesterday": compute_psi(station_ref["ratio_vs_yesterday"], recent_ratio)}
    max_feature = max(details, key=details.get)
    return max_feature, details[max_feature], details


def _rolling_wape_24h(conn, station_id: str) -> float | None:
    with conn.cursor() as cur:
        cur.execute(
            "select avg(wape) from validation_metrics "
            "where station_id = %s and computed_at >= now() - interval '24 hours'",
            (station_id,),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None


def evaluate_stations(conn, run_id: str | None, cycle_id: str | None) -> list[dict]:
    """Runs the per-station checks, logs one drift_events row per station,
    and returns the ones whose performance drift is CONFIRMED this cycle
    (each with its own drift_event id -- the row that completed the
    streak, for retrain_triggers.drift_event_id to point at)."""
    active = db.get_active_model(conn)
    if not active or not active.get("model_version", "").startswith("catboost-"):
        print("drift: no CatBoost champion active yet, skipping drift checks (baseline has no reference).")
        return []

    psi_reference = active.get("psi_reference") or {}
    per_station_valid_accuracy = (active.get("metrics_summary") or {}).get("candidate_per_station_accuracy") or {}
    champion_version = active["model_version"]

    obs_df = features.load_observations_df(conn)
    confirmed = []

    for sid in sorted(obs_df.station_id.unique()):
        series = obs_df[obs_df.station_id == sid].set_index("observed_at")["demand"]
        series.index = pd.to_datetime(series.index, utc=True)
        series = series.sort_index()

        station_ref = psi_reference.get(sid, {})
        psi_feature, psi_value, psi_details = _station_psi(series, station_ref) if station_ref else (None, 0.0, {})
        psi_flag = psi_value > PSI_FLAG_THRESHOLD

        champion_acc = per_station_valid_accuracy.get(sid)
        rolling_wape = _rolling_wape_24h(conn, sid)
        performance_flag = False
        if champion_acc is not None and rolling_wape is not None:
            rolling_accuracy = 100 * max(0.0, 1 - rolling_wape)
            performance_flag = (champion_acc - rolling_accuracy) >= PERFORMANCE_DROP_THRESHOLD

        # Confirm against the last (CONFIRM_STREAK - 1) prior checks, so
        # this cycle's own flag plus that history together decide the streak.
        prior_flags = db.get_recent_performance_flags(conn, sid, CONFIRM_STREAK - 1)
        streak = [performance_flag] + prior_flags
        performance_confirmed = len(streak) >= CONFIRM_STREAK and all(streak)

        event_id = db.log_drift_event(
            conn, run_id=run_id, cycle_id=cycle_id, station_id=sid, champion_version=champion_version,
            psi_flag=psi_flag, psi_max_feature=psi_feature, psi_max_value=psi_value, psi_details=psi_details,
            performance_flag=performance_flag, performance_confirmed=performance_confirmed,
            rolling_wape_24h=rolling_wape, champion_valid_accuracy=champion_acc,
            n_cycles_confirming=sum(1 for f in streak if f),
        )

        if psi_flag or performance_flag:
            print(f"drift: station {sid} -- psi_flag={psi_flag} (max {psi_feature}={psi_value:.3f}) "
                  f"performance_flag={performance_flag} confirmed={performance_confirmed} "
                  f"(rolling_wape_24h={rolling_wape}, champion_acc={champion_acc})")

        if performance_confirmed:
            confirmed.append({
                "station_id": sid,
                "drift_event_id": event_id,
                "rolling_wape_24h": rolling_wape,
                # Virtual-clock timestamp, not wall-clock trained_at --
                # observations.observed_at lives on the competition's
                # virtual clock, a completely different timeline.
                "training_data_end": (active.get("metrics_summary") or {}).get("training_data_end"),
            })

    return confirmed


def guards_pass(conn, confirmed: list[dict]) -> tuple[bool, str]:
    last_trigger = db.get_last_retrain_trigger_time(conn)
    if last_trigger is not None:
        elapsed_hours = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_trigger)).total_seconds() / 3600
        if elapsed_hours < COOLDOWN_HOURS:
            return False, f"cooldown active ({elapsed_hours:.1f}h since last dispatch, need {COOLDOWN_HOURS}h)"

    training_data_end = confirmed[0]["training_data_end"]
    if training_data_end is None:
        return False, "champion has no recorded training_data_end (trained before this guard existed); retrain manually once to backfill it"
    new_obs = db.count_new_observations(conn, training_data_end)
    min_needed = MIN_NEW_OBSERVATIONS * len(confirmed)
    if new_obs < min_needed:
        return False, f"not enough new data yet ({new_obs} new observations, need {min_needed})"

    return True, "ok"


def maybe_trigger_retrain(conn, confirmed: list[dict]) -> None:
    if not confirmed:
        return

    ok, reason = guards_pass(conn, confirmed)
    stations = sorted(c["station_id"] for c in confirmed)
    if not ok:
        print(f"drift: performance drift confirmed on {stations} but NOT retraining yet -- {reason}")
        return

    # Point at the confirming event of the station with the worst rolling
    # WAPE -- the single row that best explains "why this fired". Picked
    # locally from `confirmed` (already has everything needed) rather than
    # re-querying drift_events -- an earlier version did
    # `where id = any(%s)` with a list of UUID strings, which Postgres
    # rejects as "operator does not exist: uuid = text" without an explicit
    # cast. That exception propagated out of the whole `with db.connect()`
    # block in monitor.py, silently rolling back everything else done in
    # the same transaction (validation_metrics, the leaderboard snapshot,
    # every drift_events row for that cycle) -- caught live: the dashboard
    # sat frozen for over a day before this was found.
    primary_event_id = max(confirmed, key=lambda c: c["rolling_wape_24h"] or 0)["drift_event_id"]

    reason_text = f"confirmed performance drift on {stations} for {CONFIRM_STREAK}+ consecutive cycles"
    trigger_id = db.insert_retrain_trigger(conn, primary_event_id, stations, reason_text)
    print(f"drift: {reason_text} -- dispatching train-on-drift (trigger {trigger_id})")

    env = {**os.environ, "GH_TOKEN": os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))}
    result = subprocess.run(
        ["gh", "workflow", "run", "train-on-drift",
         "-f", f"stations={','.join(stations)}",
         "-f", f"trigger_id={trigger_id}"],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"drift: failed to dispatch train-on-drift: {result.stderr}")
    else:
        print("drift: train-on-drift dispatched successfully")


def run(conn, run_id: str | None, cycle_id: str | None) -> None:
    if conn is None:
        return
    confirmed = evaluate_stations(conn, run_id, cycle_id)
    maybe_trigger_retrain(conn, confirmed)
