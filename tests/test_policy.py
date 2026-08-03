"""Tests for calibra/policy.py (`calibra integrity --policy` file loading)."""

from __future__ import annotations

import json

import pytest

from calibra.policy import load_policy


def _write(tmp_path, name: str, content) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(content))
    return str(path)


class TestLoadPolicy:
    def test_valid_policy_round_trips(self, tmp_path):
        path = _write(
            tmp_path,
            "policy.json",
            {"camera_freeze_events": "block", "ldlj": "inspect"},
        )
        policy = load_policy(path)
        assert policy == {"camera_freeze_events": "block", "ldlj": "inspect"}

    def test_empty_policy_is_valid(self, tmp_path):
        path = _write(tmp_path, "policy.json", {})
        assert load_policy(path) == {}

    def test_invalid_action_raises(self, tmp_path):
        path = _write(tmp_path, "policy.json", {"camera_freeze_events": "delete"})
        with pytest.raises(ValueError, match="invalid action"):
            load_policy(path)

    def test_non_dict_top_level_raises(self, tmp_path):
        path = _write(tmp_path, "policy.json", ["block", "inspect"])
        with pytest.raises(ValueError, match="JSON object"):
            load_policy(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            load_policy(str(tmp_path / "does_not_exist.json"))
