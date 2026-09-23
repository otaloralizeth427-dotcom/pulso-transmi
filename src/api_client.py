import requests

from config import PTM_API_BASE, PTM_API_KEY


class PulsoTransmiClient:
    def __init__(self, base_url: str = PTM_API_BASE, api_key: str | None = PTM_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

    def auth_headers(self) -> dict:
        if not self.api_key:
            return {}
        return {"authorization": f"Bearer {self.api_key}"}

    def clock(self) -> dict:
        r = self.session.get(f"{self.base_url}/v1/clock", timeout=20)
        r.raise_for_status()
        return r.json()

    def current_cycle(self) -> dict | None:
        r = self.session.get(f"{self.base_url}/v1/forecast-cycles/current", timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def observations_page(self, station_id: str | None = None, start: str | None = None,
                           cursor: str | None = None, limit: int = 5000) -> dict:
        params = {"limit": limit}
        if station_id:
            params["station_id"] = station_id
        if start:
            params["start"] = start
        if cursor:
            params["cursor"] = cursor
        r = self.session.get(f"{self.base_url}/v1/observations", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def stream_observations_page(self, cursor: str | None = None, limit: int = 5000) -> dict:
        """Incremental observations released as the competition clock advances.

        Distinct from /v1/observations, which only ever serves the fixed
        starter history. Missing this distinction produced an all-zero first
        submission during the first manual run of this pipeline.
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = self.session.get(f"{self.base_url}/v1/stream/observations", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def submit(self, payload: dict, idempotency_key: str) -> tuple[int, dict]:
        headers = {**self.auth_headers(), "Idempotency-Key": idempotency_key}
        r = self.session.post(f"{self.base_url}/v1/submissions", json=payload, headers=headers, timeout=30)
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text}
        return r.status_code, body

    def leaderboard(self, window: str = "cumulative") -> dict:
        r = self.session.get(f"{self.base_url}/v1/leaderboard", params={"window": window},
                              headers=self.auth_headers(), timeout=20)
        r.raise_for_status()
        return r.json()
