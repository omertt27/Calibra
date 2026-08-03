"""
Integrity CI policy files.

A policy is a flat JSON object mapping a metric name (the same names shown
in `calibra integrity --json`, e.g. "camera_freeze_events") to the action a
CRITICAL finding on that metric should take:

    "block"    fail CI (exit 1)
    "inspect"  surface the finding, don't fail the build

Metrics not listed keep `calibra integrity`'s built-in default (see
`calibra/integrity.py`'s `_MOTION_REVIEW_METRICS`). OK and WARNING findings
are unaffected by policy — only CRITICAL findings' CI consequence is
configurable.

This lets different teams encode different risk tolerances against the same
diagnostics — e.g. a research lab that only wants to block on corrupted
timestamps, vs. a production team that also blocks on camera freezes, vs. a
team that's validated calibration-drift thresholds enough to block on those
too.

Example:
    {
      "timestamp_jitter_cv": "block",
      "camera_freeze_events": "block",
      "joint_offset_max_abs": "inspect",
      "jerk_spike_rate": "inspect"
    }
"""

from __future__ import annotations

import json

_VALID_ACTIONS = {"block", "inspect"}


def load_policy(path: str) -> dict[str, str]:
    """
    Load and validate a policy file. Raises ValueError on malformed content;
    file-not-found and other I/O errors propagate as-is.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"policy file must be a JSON object of {{metric: action}}, got {type(data).__name__}"
        )

    invalid = {k: v for k, v in data.items() if v not in _VALID_ACTIONS}
    if invalid:
        raise ValueError(
            f"invalid action(s) in policy file (must be 'block' or 'inspect'): {invalid}"
        )

    return data
