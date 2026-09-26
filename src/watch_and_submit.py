"""Polls for open cycles for the whole run instead of exiting after the first
catch, so one triggered run can cover an entire hour of cycles on its own.

Why this exists: GitHub's free-tier cron is unreliable -- it silently drops
or delays scheduled runs under load. Across the first 47 hours of this
pipeline, only 12 of ~94 expected half-hourly triggers actually fired. A
first version of this script polled for ~12 minutes and exited on its first
successful submission; that still left most of every hour uncovered whenever
a trigger didn't fire again soon after. This version stays alive for most of
an hour and submits to every cycle that opens during that window, so a
single trigger firing anywhere in an hour is enough to cover that hour.

`pipeline.yml` also runs this with `concurrency: cancel-in-progress: false`,
so if a new trigger fires while a run is still mid-poll, it queues instead
of killing the run that might be seconds from catching an open cycle --
consecutive runs chain into continuous coverage rather than competing.
"""
import sys
import time

import ingest
import monitor
import predict
from api_client import PulsoTransmiClient

POLL_BUDGET_SECONDS = 50 * 60  # just under GitHub's 55-minute job timeout,
                                # so one run can span nearly a full hour.
POLL_INTERVAL_SECONDS = 30


def run() -> int:
    client = PulsoTransmiClient()
    deadline = time.monotonic() + POLL_BUDGET_SECONDS
    attempt = 0
    submitted_cycles: set[str] = set()

    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"watch: time budget exhausted after {attempt} attempt(s); "
                  f"caught {len(submitted_cycles)} cycle(s) this run: {sorted(submitted_cycles)}")
            break

        cycle = client.current_cycle()
        cycle_id = cycle.get("cycle_id") if cycle else None
        is_open = cycle is not None and cycle.get("state") == "open"

        if is_open and cycle_id in submitted_cycles:
            pass  # already handled this one; just wait for it to close or for a new one
        elif is_open:
            print(f"watch: cycle {cycle_id} is open (attempt {attempt}) -- running ingest + predict")
            try:
                ingest.run()
            except Exception as exc:
                print(f"watch: ingest failed this attempt, will retry: {exc}")
            try:
                result = predict.run()
            except Exception as exc:
                print(f"watch: predict failed this attempt, will retry: {exc}")
                result = 1
            if result == 0:
                print(f"watch: submission handled successfully for {cycle_id}")
                submitted_cycles.add(cycle_id)
            else:
                print("watch: predict did not succeed this attempt, will keep polling")

        time.sleep(min(POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))

    try:
        monitor.run()
    except Exception as exc:
        print(f"watch: monitor failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
