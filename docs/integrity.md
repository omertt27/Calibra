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
     episode ep_17), suggesting the camera stopped updating.
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
──────────────────────────────────────────────────────────
```

Exit code is `1` if any check is CRITICAL, `0` otherwise — safe to wire into a pre-training CI gate.

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
