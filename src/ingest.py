"""Collector: pulls new observations from the incremental stream and
upserts them into Supabase, advancing the cursor only after a successful
commit. Safe to run every 30 minutes and safe to run twice in a row --
running it with no new data is a normal, successful exit, not an error.
"""
import sys

import db
from api_client import PulsoTransmiClient

SOURCE = "stream_observations"


def run() -> int:
    client = PulsoTransmiClient()
    with db.connect() as conn:
        cursor = db.get_ingestion_cursor(conn, SOURCE)
        resume_cursor = cursor  # last position known to have more data after it
        total = 0
        last_observed_at = None
        pages = 0
        while True:
            page = client.stream_observations_page(cursor=cursor, limit=5000)
            rows = page["data"]
            if rows:
                db.upsert_observations(conn, rows)
                total += len(rows)
                last_observed_at = rows[-1]["observed_at"]
            next_cursor = page.get("next_cursor")
            pages += 1
            if not next_cursor or next_cursor == cursor:
                # Caught up to the current frontier: null next_cursor means
                # "nothing more right now". If we'd already advanced past a
                # non-null cursor this run, keep that position for next time
                # instead of losing it. If the whole available stream fit in
                # one page (no non-null cursor was ever issued), there is no
                # resume position to persist -- the next run re-requests from
                # the start. That's wasteful once the stream grows large, but
                # harmless: upsert on (station_id, observed_at) makes
                # re-ingesting already-seen rows a no-op, never a duplicate.
                break
            resume_cursor = next_cursor
            cursor = next_cursor
            if pages > 100:
                break
        db.set_ingestion_cursor(conn, SOURCE, resume_cursor, last_observed_at)
        print(f"ingest: {total} new observations across {pages} page(s); "
              f"cursor now at {resume_cursor!r}, last_observed_at={last_observed_at}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
