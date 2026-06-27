"""calibra.cloud.auth — token storage and identity for Calibra Cloud."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

_CONFIG_PATH = Path.home() / ".calibra" / "config.json"
_CLOUD_URL = "https://app.calibra.io"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_token() -> str | None:
    return _load_config().get("api_token")


def is_authenticated() -> bool:
    return bool(get_token())


def _call_me(token: str) -> dict | None:
    try:
        req = urllib.request.Request(
            f"{_CLOUD_URL}/api/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def run_login() -> None:
    print(f"Log in to Calibra Cloud — {_CLOUD_URL}")
    print(f"Get your API token at: {_CLOUD_URL}/settings/tokens")
    print()
    try:
        token = input("Paste your API token: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not token:
        print("No token entered.")
        return

    user = _call_me(token)
    config = _load_config()
    config["api_token"] = token
    _save_config(config)

    if user:
        email = user.get("email", "unknown")
        plan = user.get("plan", "free")
        print(f"✓ Logged in as {email} ({plan} plan)")
    else:
        # Cloud not live yet — store token for when it launches
        print("✓ Token saved.")
        print(
            "  Calibra Cloud is not yet live. Your token will activate automatically at launch.\n"
            f"  Visit {_CLOUD_URL} to join the waitlist."
        )


def run_whoami() -> None:
    token = get_token()
    if not token:
        print("Not logged in. Run `calibra login`.")
        return

    user = _call_me(token)
    if user:
        email = user.get("email", "unknown")
        plan = user.get("plan", "free")
        projects = user.get("project_count", 0)
        print(f"{email} · {plan} plan · Projects: {projects}")
    else:
        print(f"Token stored (cloud not yet live). Visit {_CLOUD_URL}")


def run_logout() -> None:
    config = _load_config()
    if "api_token" not in config:
        print("Not logged in.")
        return
    del config["api_token"]
    _save_config(config)
    print("✓ Logged out.")
