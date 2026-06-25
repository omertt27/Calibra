"""
Tests for calibra.schema.normalization — key mapping and resolution rules.
"""

from __future__ import annotations

import warnings

import numpy as np

from calibra.schema.normalization import normalize_obs_keys, _resolve, _DEFAULT_MAPPING


# ── helpers ───────────────────────────────────────────────────────────────────


def _arr(shape=(10, 3)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


# ── exact-match rules ─────────────────────────────────────────────────────────


def test_exact_match_state():
    result = normalize_obs_keys({"state": _arr()})
    assert "state" in result


def test_exact_match_lerobot_image_top():
    result = normalize_obs_keys({"images.top": _arr()})
    assert "camera_top" in result
    assert "images.top" not in result


def test_exact_match_lerobot_image_wrist():
    result = normalize_obs_keys({"images.wrist": _arr()})
    assert "camera_wrist" in result


def test_exact_match_robomimic_eef_pos():
    result = normalize_obs_keys({"robot0_eef_pos": _arr()})
    assert "eef_position" in result


def test_exact_match_robomimic_joint_pos():
    result = normalize_obs_keys({"robot0_joint_pos": _arr()})
    assert "joint_position" in result


def test_exact_match_joint_velocity():
    result = normalize_obs_keys({"joint_vel": _arr()})
    assert "joint_velocity" in result


def test_exact_match_proprio():
    result = normalize_obs_keys({"proprio": _arr()})
    assert "state" in result


# ── prefix-strip rules ────────────────────────────────────────────────────────


def test_prefix_strip_images_unknown_camera():
    result = normalize_obs_keys({"images.front_left": _arr()})
    assert "camera_front_left" in result


def test_prefix_strip_image_dot():
    result = normalize_obs_keys({"image.overhead_cam": _arr()})
    assert "camera_overhead_cam" in result


def test_prefix_strip_obs_slash():
    result = normalize_obs_keys({"obs/some_sensor": _arr()})
    assert "some_sensor" in result


def test_prefix_strip_robot0():
    # robot0_custom → strips robot0_, remainder 'custom' not in mapping → pass-through
    result = normalize_obs_keys({"robot0_custom_sensor": _arr()})
    assert "custom_sensor" in result


# ── pass-through ──────────────────────────────────────────────────────────────


def test_unknown_key_passes_through():
    result = normalize_obs_keys({"completely_custom_key_xyz": _arr()})
    assert "completely_custom_key_xyz" in result


def test_empty_dict():
    assert normalize_obs_keys({}) == {}


# ── extra_mapping override ────────────────────────────────────────────────────


def test_extra_mapping_overrides_default():
    # By default "state" → "state"; override to "joint_state"
    result = normalize_obs_keys(
        {"state": _arr()},
        extra_mapping={"state": "joint_state"},
    )
    assert "joint_state" in result
    assert "state" not in result


def test_extra_mapping_adds_new_rule():
    result = normalize_obs_keys(
        {"my_lidar": _arr()},
        extra_mapping={"my_lidar": "lidar_scan"},
    )
    assert "lidar_scan" in result


def test_extra_mapping_does_not_affect_other_keys():
    result = normalize_obs_keys(
        {"images.top": _arr(), "state": _arr()},
        extra_mapping={"my_key": "other"},
    )
    assert "camera_top" in result
    assert "state" in result


# ── duplicate canonical key warning ──────────────────────────────────────────


def test_duplicate_canonical_key_emits_warning():
    # "proprio" → "state" and "state" → "state" — both map to "state"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_obs_keys({"state": _arr(), "proprio": _arr()})
        assert any("state" in str(warning.message) for warning in w)
    assert "state" in result


# ── arrays are not copied ─────────────────────────────────────────────────────


def test_arrays_are_same_object():
    arr = _arr()
    result = normalize_obs_keys({"images.top": arr})
    assert result["camera_top"] is arr


# ── _resolve directly ────────────────────────────────────────────────────────


def test_resolve_exact():
    assert _resolve("images.top", _DEFAULT_MAPPING) == "camera_top"


def test_resolve_prefix():
    assert _resolve("images.new_camera", _DEFAULT_MAPPING) == "camera_new_camera"


def test_resolve_passthrough():
    assert _resolve("totally_unknown", _DEFAULT_MAPPING) == "totally_unknown"


# ── multi-key batch ───────────────────────────────────────────────────────────


def test_mixed_batch_normalizes_all():
    raw = {
        "images.top": _arr(),
        "images.wrist": _arr((10, 3)),
        "state": _arr((10, 14)),
        "custom_sensor": _arr((10, 1)),
    }
    result = normalize_obs_keys(raw)
    assert set(result.keys()) == {"camera_top", "camera_wrist", "state", "custom_sensor"}
