# Integrity Checks

`calibra integrity` answers the first question practitioners ask about a new dataset — before quality, diversity, or optimization matter at all:

> **Can I trust this dataset?**

It runs a fixed, cheap set of checks and groups findings into **Critical / Warnings / Passed**, with a demoted `Integrity Score` as a summary line rather than the headline result — a finding is never collapsed straight to a single number.

```bash
calibra integrity /data/my_demos.h5
calibra integrity /data/my_demos.h5 --format hdf5
calibra integrity /data/my_demos.h5 --json
```

```
─── Dataset Integrity ────────────────────────────────────
my_demos · 120 episodes

Critical (1)
  ❌ camera_freeze_events: 1 of 120 episodes (0.8%) contain a run of ≥5
     consecutive near-identical camera frames (longest: 14 frames in
     episode ep_17), suggesting the camera stopped updating.  [BLOCK]
      A frozen camera segment means the policy would be trained on stale
      visual input during that window — likely to cause visually-triggered
      failures when deployed against a live, moving scene.

Warnings (0)

Passed (8)
  ✅ timestamp_jitter_cv: Step-to-step timing is consistent.
  ✅ timestamp_dropout_rate: Timestamp dropout rate is within acceptable range.
  ✅ action_dropout_rate: Action-dropout rate is within acceptable range.
  ✅ short_episode_fraction: No suspiciously short episodes detected.
  ✅ duplicate_frame_rate: Camera frames show expected frame-to-frame variation.
  ✅ ldlj: Action trajectories are smooth (LDLJ within threshold).
  ✅ jerk_spike_rate: Jerk spike rate is within acceptable range.
  ✅ velocity_discontinuity_rate: Velocity profile is continuous — no sudden reversals.

Integrity Score: 89/100  ·  Status: Warning
CI result: Failed  ·  Reason: 1 CRITICAL block-level finding(s): camera_freeze_events
──────────────────────────────────────────────────────────
```

`Status` and `CI result` are deliberately separate. `Status` is a severity summary across every finding (can read "Warning" even with a CRITICAL present, if enough other checks passed). `CI result` — the thing that actually sets the exit code — only fails on **block**-level findings: objective acquisition/format/sync/completeness failures (dropped timestamps, missing frames, a frozen or corrupted camera, incomplete episodes). Findings whose `suggested_action` is `inspect` — the three motion-smoothness metrics (`ldlj`, `jerk_spike_rate`, `velocity_discontinuity_rate`), which can legitimately hit CRITICAL on a scripted/planner-collected dataset — are a review signal, not proof the dataset is broken, and don't fail CI on their own.

```bash
calibra integrity /data/my_demos.h5 --strict
```

Pass `--strict` to fall back to the old, blunter policy: fail on *any* CRITICAL finding, including motion-review ones. Every finding also carries a `suggested_action` (`informational` / `inspect` / `block`) in `--json` output, so a CI pipeline that wants a custom policy can compute its own pass/fail from the raw findings instead of relying on the built-in `ci_result`.

An analyzer that couldn't run at all (e.g. duplicate-frame/camera-freeze/blur on a video-backed LeRobot v2/v3 dataset without `--decode-images`) shows up under **Not Evaluated** with a reason, rather than disappearing from the output silently.

### CI Policy Files

The built-in block/inspect split is one reasonable default, not the only one a team might want. A research lab might only want to block on corrupted timestamps; a production team might also block on camera freezes; another team might treat calibration drift as a blocker once they've validated the thresholds against their own hardware. `--policy` lets you configure that per metric, instead of the all-or-nothing choice between the default and `--strict`:

```bash
calibra integrity /data/my_demos.h5 --policy ci_policy.json
```

```json
{
  "timestamp_jitter_cv": "block",
  "camera_freeze_events": "block",
  "joint_offset_max_abs": "inspect",
  "jerk_spike_rate": "inspect"
}
```

A policy file is a flat JSON object mapping a **metric name** — the exact names shown in the `metric` field of `--json` output, e.g. `camera_freeze_events`, `blurry_episode_fraction`, `joint_offset_max_abs` — to `"block"` or `"inspect"`. It only overrides the CI consequence of a **CRITICAL** finding on that metric; OK and WARNING findings are unaffected — a WARNING never fails CI, policy or not. Metrics not listed keep the built-in default. An unrecognized metric name in the file prints a warning to stderr and is otherwise ignored, so a policy written against an older or newer Calibra version doesn't hard-fail.

`--policy` and `--strict` are mutually exclusive — once you're configuring per-metric behavior, that fully replaces the blunt "fail on anything" fallback rather than the two combining in some precedence order.

---

## Checks

### Timestamp consistency

**Detects:** irregular step-to-step timing (`timestamp_jitter_cv`) and dropped/missed ticks (`timestamp_dropout_rate`), via coefficient of variation and gap-vs-median-delta analysis.

**Why it matters:** irregular control-loop timing degrades time-series policies (transformers, diffusion) that assume fixed-frequency data. Dropout creates artificial velocity discontinuities that BC policies learn as spurious high-jerk transitions.

**Example output:**
```
⚠️  timestamp_jitter_cv: High coefficient of variation in inter-step timing
    (18.3% mean CV across 120 episodes).
     Irregular control-loop timing degrades time-series policies that
     assume fixed-frequency data. Consider resampling to a uniform
     frequency before training.
```

### Sensor synchronization

**Detects:** camera-timestamp lag relative to the master clock (`camera_lag_std[<camera>]`), action/observation timestamp misalignment (`action_obs_misalignment`), and camera-to-proprioception render lag (`camera_physics_drift`, primarily relevant to simulated data).

**Why it matters:** closed-loop policies that fuse camera and proprioception observations will experience systematic desync, especially around contact transitions where precise timing matters most.

### Episode completeness

**Detects:** episodes with a step count far below the dataset's typical length (`short_episode_fraction`, IQR-outlier test).

**Why it matters:** short-outlier episodes are usually failed demonstrations that were never filtered out — an operator abort, a safety-limit trip, or early termination. Training on them teaches the policy to abort tasks prematurely.

### Duplicate frames

**Detects:** camera frames that are near-identical to the frame immediately before them (`duplicate_frame_rate`) — a dropped grab or re-emitted buffer rather than a genuinely new observation.

**Why it matters:** a policy trained on repeated frames can learn to associate a stale visual observation with the wrong action, or waste model capacity encoding redundant input.

**Example output:**
```
⚠️  duplicate_frame_rate: 7.2% of camera frame transitions are near-identical
    to the previous frame, across 118 episodes with image data.
     Duplicate frames mean the camera pipeline is not capturing a new
     image every control step.
```

### Camera freeze

**Detects:** a *sustained run* (default: 5+ consecutive frames) of near-identical camera frames within an episode — the camera stopped updating entirely, not just a single dropped grab (`camera_freeze_events`).

**Why it matters:** a frozen camera segment trains the policy on stale visual input for that whole window, which shows up as visually-triggered failures once deployed against a real, moving scene. This is a stronger signal than an isolated duplicate frame, which is why it's tracked as a separate, more severe check.

### Blur

**Detects:** episodes whose camera frames are anomalously blurry *relative to the rest of the dataset* (`blurry_episode_fraction`), via an IQR-outlier test on per-episode mean Laplacian variance — the standard, dependency-free sharpness proxy.

**Why it matters:** motion blur, defocus, or a dirty/misconfigured lens gives the policy a degraded or misleading visual observation for that episode.

**Example output:**
```
⚠️  blurry_episode_fraction: 14.3% of episodes (1/7) have camera frames
    markedly blurrier than the rest of the dataset (mean Laplacian
    variance below the IQR lower fence). IDs: ['episode_3']
     Inspect the flagged episodes before training.
```

!!! note "Relative, not absolute"
    Laplacian variance has no universal "blurry" cutoff — it depends on camera resolution, exposure, and scene content. This check flags episodes that are outliers *within their own dataset*, the same statistical approach used for episode completeness above. A dataset that's uniformly blurry throughout won't self-flag; there's nothing anomalous to compare against.

!!! note "Format coverage"
    Duplicate-frame, camera-freeze, and blur detection all need decoded image arrays. They work on HDF5-format datasets (Isaac Lab, robomimic-style exports) by default, and on **LeRobot v1** datasets (HuggingFace `Image`-feature columns) via the opt-in `--decode-images` flag — off by default since decoding increases load time and memory use. **LeRobot v2/v3** datasets store frames as encoded mp4 video rather than per-frame images; decoding those isn't supported yet (deliberately deferred — v3 in particular stores multiple episodes concatenated into shared video files with per-episode timestamp offsets, which needs more validation against real data before shipping). On v2/v3, these three checks are silently skipped (not an error), same as before.

### Jittery / jerky motion

**Detects:** physically jittery recorded motion via three metrics — smoothness (`ldlj`, Logarithmic Dimensionless Jerk), abrupt jerk spikes (`jerk_spike_rate`), and sudden velocity reversals (`velocity_discontinuity_rate`).

**Why it matters:** high jerk in demonstration data forces the policy to learn discontinuous action transitions; BC policies trained on jerky data produce jerky rollouts that stress hardware and reduce task success, especially on contact-rich tasks.

**Example output:**
```
⚠️  ldlj: Mean LDLJ = -12.40 across 118 episodes (threshold: >-10).
    Action trajectories contain significant jerk.
     High jerk in demonstration data forces the policy to learn
     discontinuous action transitions. Consider applying action
     smoothing (e.g. Savitzky-Golay) before training.
```

!!! note "Integrity vs. Quality split for motion"
    `calibra integrity` only checks whether the recorded motion is *physically jittery* — the same three metrics regardless of why. It does not check action-state tracking error or scripted-vs-teleoperated collection signature; those are collection-method questions, not a trust check, and stay under `calibra audit` (Motion Quality dimension).

### Leader/follower calibration drift

**Detects:** a systematic, consistent per-motor offset between the commanded action and the observed joint state (`action - observation.state`), measured only during sustained stationary "hold" frames — not during motion, where the gap is dominated by ordinary tracking lag rather than a calibration issue (`joint_offset_max_abs`).

**Why it matters:** a stale leader/follower calibration — e.g. the arms were re-zeroed differently, or a joint offset drifted between sessions — produces exactly this signature: a *stable*, motor-specific bias that's invisible in training loss (the policy just learns the biased mapping) but shows up as consistent under/overshoot at deployment. This is the failure mode described in [LeRobot issue #3758](https://github.com/huggingface/lerobot/issues/3758): an unnoticed ~17° offset on one joint produced roughly a 6cm Cartesian gripper error. The check is post-hoc and read-only — it uses `action` and a paired state observation already present in most teleoperation datasets, no new data collection required.

**Example output:**
```
⚠️  joint_offset_max_abs: Motor/dim 2 shows a consistent action-state offset
    of 0.31 (12.4% of its observed range) across 340 pooled stationary
    frames, stable across holds (|mean|/std = 4.2).  [INSPECT]
     A stable per-motor action/state offset during holds — not during
     motion — is the signature of a leader/follower calibration drift.
     This is a review signal, not proof of a broken dataset: verify the
     leader/follower zero-offset calibration before relying on this data
     for precision manipulation.
```

!!! note "Uncalibrated thresholds — capped at WARNING"
    Unlike `action_state_divergence` (calibrated from 12 reference hardware profiles), this check has not yet been validated against a dataset with a known injected offset. It never reports CRITICAL, and `suggested_action` is always `inspect`, until real-world calibration data is available. The offset threshold is expressed as a fraction of each motor's own observed range (not an absolute unit like radians), since state units vary by dataset and have no universal scale — the same reasoning the [Blur](#blur) check applies to Laplacian variance.

---

## Demo recording

`docs/demo.tape` is a [VHS](https://github.com/charmbracelet/vhs) script that reproduces the full workflow end-to-end — `calibra integrity` on a small synthetic dataset (generated by `docs/demo_fixture.py`, no external data needed), then a peek at `audit`/`review`/`prune`. Generate the GIF locally with:

```bash
brew install vhs
vhs docs/demo.tape   # writes docs/figures/integrity_demo.gif
```

---

## Next steps

Once a dataset passes Integrity, move to:

- `calibra audit` / `calibra score` — is the data *clean*? (action-state tracking error, scripted-vs-teleop signature, task structure)
- `calibra review` — is it *diverse* enough, and which episodes are worth a human look?
- `calibra prune` — build a smaller, high-quality coreset for training

See the [full command reference](commands.md) for all of the above.
