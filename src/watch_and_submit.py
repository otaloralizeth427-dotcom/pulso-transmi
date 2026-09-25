"""Polls for an open cycle instead of checking once and giving up.

Why this exists: GitHub's free-tier cron is unreliable at 30-minute
granularity -- it silently drops or delays scheduled runs under load. Across
the first 47 hours of this pipeline, only 12 of the ~94 expected half-hourly
triggers actually fired, and several of those landed outside any cycle's
25-minute window. The fix isn't a tighter cron (GitHub throttles frequent
schedules even harder) -- it's making each run that DOES fire stay alive and
keep checking for a while, so it doesn't need to land on the exact minute.

Exits as soon as a submission succeeds, or when the time budget runs out.
"""
import sys
import time

import ingest
import monitor
import predict
from api_client import PulsoTransmiClient

POLL_BUDGET_SECONDS = 12 * 60  # a bit more than the 10-minute cron interval,
                                # so consecutive triggers overlap and cover
                                # the gaps GitHub's scheduler leaves.
POLL_INTERVAL_SECONDS = 30


def run() -> int:
    client = PulsoTransmiClient()
    deadline = time.monotonic() + POLL_BUDGET_SECONDS
    attempt = 0

    while True:
        attempt += 1
        cycle = client.current_cycle()
        if cycle is not None and cycle.get("state") == "open":
            print(f"watch: cycle {cycle['cycle_id']} is open (attempt {attempt}) -- running ingest + predict")
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
                print("watch: submission handled successfully")
                break
            print("watch: predict did not succeed this attempt, will keep polling")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"watch: time budget exhausted after {attempt} attempt(s), no open cycle caught this run")
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

    try:
        monitor.run()
    except Exception as exc:
        print(f"watch: monitor failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
