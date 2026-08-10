# Changelog

All notable changes to Calibra are documented here.

## [0.8.0] — Measured training results

Calibra's predictions (`calibra benchmark`) were always simulated — a heuristic outcome model plus linear GPU-hour scaling. That's useful for a first pass, but not evidence. This adds the other half: a way to record what a design partner's *real* training runs actually cost and achieved, and to fold those measured numbers into the benchmark report wherever they're available, so a report is never presented as a validated result when parts of it are still predictions.

### Added

- **`calibra experiment record`** — logs one training run's result (GPU-hours, wall-clock time, energy, eval success rate) against the design-partner protocol's `full` / `random` / `calibra` conditions at a given retention percentage. Training itself runs in the partner's own pipeline; this only records the outcome. Stored locally as JSON Lines at `~/.calibra/experiments.jsonl` — never synced to any network endpoint. New module `calibra/experiment_log.py`.
- **`calibra experiment list` / `calibra experiment report`** — list recorded runs, or print the full retention-curve comparison for one experiment, including the Calibra-vs-random delta at each level and which `(retention%, condition)` pairs the protocol still expects but haven't been recorded yet. New CLI handler `calibra/experiment.py`.
- **`calibra benchmark --sweep`** — runs the full design-partner retention curve (default `10/25/50/75/100%`, override with `--fractions`) in one shot instead of a single `--keep` fraction.
- **`calibra benchmark --experiment-id ID`** — substitutes real measured GPU-hours / eval success rate from `calibra experiment record` into the benchmark report wherever a matching condition and retention level has been logged, falling back to simulated values for anything not yet measured.
- Benchmark reports now carry a **status**: `SIMULATED` (nothing measured — a prediction), `PARTIAL MEASUREMENT` (some conditions measured, others still simulated — not safe to report as validated), or `CASE STUDY / VALIDATED` (full, random, and Calibra all backed by real recorded training runs). Every number is individually tagged `(measured)` or `(simulated)`.
- Compute savings are now computed from GPU-hours rather than episode-count reduction, since the two can diverge once real measured numbers are mixed in.

See `docs/commands.md`'s `calibra benchmark` and `calibra experiment` sections for the full partner workflow.

### Fixed

- `compute_trajectory_entropy` force-cast actions to `float32` before `np.histogram`, which could collapse distinct values (and raise "Too many bins for data range") on large-offset, sub-millimeter-variance action columns — exactly the near-duplicate-trajectory case the function is meant to detect. Now computed in `float64`. Reported via a Reddit user testing on real data.

## [0.7.3] — Configurable Integrity CI policies

Direct follow-up to the same HF feedback thread that drove 0.7.2: after reviewing the built-in block/inspect split, the reviewer proposed letting teams configure it themselves rather than shipping one fixed policy for everyone (a research lab only blocking on corrupted timestamps vs. a production team also blocking on camera freezes vs. a team that's validated calibration-drift thresholds enough to block on those too).

### Added

- **`calibra integrity --policy FILE`** — a flat JSON file mapping a metric name (the exact names shown in `--json` output, e.g. `camera_freeze_events`) to `"block"` or `"inspect"`, overriding the built-in default for CRITICAL findings on the metrics it names. OK and WARNING findings are unaffected — a WARNING never fails CI regardless of policy. New module `calibra/policy.py` handles loading and validation. `--strict` and `--policy` are mutually exclusive. `--json` output gains a `policy_path` field for traceability. See `docs/integrity.md`'s "CI Policy Files" section.
- Policy files are JSON, not YAML — the core install has zero non-`numpy`/`pydantic` dependencies, and JSON needs no new one.

## [0.7.2] — CI policy split, Not Evaluated, calibration drift

Prompted by a community probe against six public LeRobot datasets (including matched ALOHA human/scripted pairs) run with `calibra integrity` v0.7.1, which surfaced a real gap: every dataset returned exit code 1, while five of six still showed overall `Status: Warning` — because a single CRITICAL motion-smoothness finding (which can legitimately fire on a scripted/planner dataset) carried the exact same operational weight as a dropped-timestamp or corrupted-frame finding.

### Added

- Every finding now carries a `suggested_action`: `informational` (OK), `inspect` (WARNING, or CRITICAL on a context-dependent motion-review metric), or `block` (CRITICAL on an objective acquisition/format/sync/completeness failure).
- `ci_result` (`Passed`/`Failed`) and `ci_reason` are now reported separately from the severity-only `status` line, and are what actually sets the exit code. By default, only `block`-level CRITICALs fail CI — `ldlj`, `jerk_spike_rate`, and `velocity_discontinuity_rate` no longer fail the build on their own.
- New `--strict` flag restores the old "any CRITICAL fails" behavior for pipelines that want the blunter policy.
- Skipped analyzers (e.g. camera checks on video-backed LeRobot v2/v3 without `--decode-images`) are now surfaced as `not_evaluated` with a reason, instead of disappearing from the output silently.
- **New check: leader/follower calibration drift** (`joint_offset_max_abs`, via `CalibrationDriftAnalyzer`) — detects a systematic per-motor `action - observation.state` offset during sustained stationary hold frames, the signature described in [LeRobot issue #3758](https://github.com/huggingface/lerobot/issues/3758) (a stable joint offset that trains fine but causes consistent under/overshoot at deployment). Post-hoc, read-only, thresholds capped at WARNING until validated against reference hardware data — see `docs/integrity.md`.

### Changed

- `calibra integrity`'s exit code now reflects `ci_result` rather than "any CRITICAL present." Existing CI pipelines that depend on the old blanket behavior should add `--strict`.

## [0.7.1] — Motion smoothness moves into Integrity

### Added

- **`calibra integrity`** now checks jittery/jerky motion — smoothness (`ldlj`), jerk spikes (`jerk_spike_rate`), and velocity discontinuities (`velocity_discontinuity_rate`) — via `ControlSmoothnessAnalyzer`. Prompted by direct practitioner feedback that named this alongside timestamps and blur as one of the recurring basics.
- `docs/integrity.md` — new "Jittery / jerky motion" section documenting the three metrics and the Integrity-vs-Quality split for motion (physical jitter is a trust check; tracking error and scripted-vs-teleop signature stay under `calibra audit`'s Motion Quality dimension).

### Fixed

- `docs/demo_fixture.py`'s synthetic actions were i.i.d. per-step noise — maximally jittery by construction, which would have swamped this release's own new checks. Replaced with a smooth low-frequency-sinusoid generator so the demo's "Passed" checks stay meaningful next to its three intentionally-injected defects (short episode, camera freeze, blur).

## [0.7.0] — Dataset Integrity

### Added

- **`calibra integrity`** — new front-door command answering "can I trust this dataset?" before quality, coverage, or optimization matter. Findings are grouped into Critical / Warnings / Passed rather than led with a single score; an `Integrity Score` is still computed but demoted to a summary line.
  - Timestamp consistency and sensor sync (jitter, dropout, camera lag, action/observation alignment)
  - Episode completeness (statistical short-episode detection)
  - Duplicate frame detection
  - Camera freeze detection
  - Blur detection
  - Image integrity checks (duplicate/freeze/blur) now available for **LeRobot v1** datasets via the new opt-in `--decode-images` flag
- `LeRobotReader(decode_images=True)` — decodes HuggingFace `Image`-feature columns for LeRobot v1 datasets. Off by default (increases load time/memory); no effect on the existing v2/v3 fast path.
- Homepage and documentation reorganized around the Integrity → Quality → Coverage → Optimize workflow (new `docs/integrity.md`, updated README, `mkdocs.yml` nav, demo assets under `docs/demo.tape`/`docs/demo_fixture.py`).
- Hugging Face Space (`spaces/app.py`) reorganized to check Dataset Integrity first, ahead of the Quality/Coverage score.

### Notes

- Duplicate-frame, camera-freeze, and blur checks work out of the box on HDF5/Isaac Lab/robomimic data, and on LeRobot v1 via `--decode-images`. LeRobot v2/v3 (video-encoded) datasets are intentionally out of scope for this release — `--decode-images` prints a warning and has no effect there.

## Planned next

- **Vision Integrity for video-backed LeRobot datasets (v2/v3)** — decode a sampled subset of frames from LeRobot's mp4-encoded v2/v3 datasets so duplicate-frame/camera-freeze/blur detection work there too. Needs a new video-decoding dependency and parsing per-episode chunk/timestamp offsets from `meta/episodes/*.parquet` (v3's multi-episode-per-file layout) — deferred until it can be validated against representative real datasets rather than shipped speculatively.
