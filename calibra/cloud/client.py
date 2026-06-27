"""calibra.cloud.client — HTTP client for the Calibra Cloud API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from calibra.cloud.auth import get_token

_DEFAULT_BASE = "https://app.calibra.io"


def _base() -> str:
    return os.environ.get("CALIBRA_CLOUD_URL", _DEFAULT_BASE)


class CalibraCloudError(Exception):
    pass


class CalibraCloudClient:
    def __init__(self) -> None:
        self.token = get_token()
        if not self.token:
            raise CalibraCloudError("Not authenticated. Run `calibra login` first.")
        self.base = _base()

    def _request(self, method: str, path: str, body: Any = None) -> dict:
        url = f"{self.base}/api{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise CalibraCloudError(f"API error {exc.code}: {exc.reason}") from exc
        except Exception as exc:
            raise CalibraCloudError(
                f"Could not reach Calibra Cloud ({self.base}). "
                "The service may not be live yet — check calibra.io for launch updates."
            ) from exc

    def push_report(self, report_json: dict, dataset_name: str) -> str:
        """Upload a DiagnosticReport. Returns the shareable dashboard URL."""
        result = self._request(
            "POST",
            "/reports",
            {"dataset_name": dataset_name, "report": report_json},
        )
        report_id = result.get("id", "")
        return result.get("url", f"{self.base}/reports/{report_id}")

    def push_outcome(self, record: dict) -> None:
        """Sync an outcome fingerprint to the authenticated outcomes endpoint."""
        self._request("POST", "/outcomes/authenticated", record)

    def get_me(self) -> dict:
        return self._request("GET", "/me")
