# Calibra Command Reference

Full documentation for all Calibra CLI commands. For a quick overview see the [README](../README.md).

---

## `calibra integrity` — "Can I trust this dataset?"

```bash
calibra integrity /data/robot_demos.h5
calibra integrity /data/robot_demos.h5 --format hdf5
calibra integrity /data/robot_demos.h5 --json
```

```
─── Dataset Integrity ────────────────────────────────────
robot_demos · 120 episodes

Critical (0)

Warnings (1)
  ⚠️  timestamp_jitter_cv: High coefficient of variation in inter-step
      timing (18.3% mean CV across 120 episodes).
      Irregular control-loop timing degrades time-series policies that
      assume fixed-frequency data.

Passed (9)
  ✅ timestamp_dropout_rate: Timestamp dropout rate is within acceptable range.
  ✅ action_dropout_rate: Action-dropout rate is within acceptable range.
  ✅ short_episode_fraction: No suspiciously short episodes detected.
  ✅ duplicate_frame_rate: Camera frames show expected frame-to-frame variation.
  ✅ camera_freeze_events: No sustained camera-freeze runs detected.
  ✅ blurry_episode_fraction: Camera frames are consistently sharp.
  ✅ ldlj: Action trajectories are smooth (LDLJ within threshold).
  ✅ jerk_spike_rate: Jerk spike rate is within acceptable range.
  ✅ velocity_discontinuity_rate: Velocity profile is continuous — no sudden reversals.

Integrity Score: 95/100  ·  Status: Healthy
──────────────────────────────────────────────────────────
```

Runs before every other command in the recommended workflow — timestamp consistency, sensor sync, episode completeness, duplicate/frozen/blurry camera frames, and jittery/jerky motion. Findings are grouped into Critical/Warnings/Passed rather than led with a single score; the score is still computed but demoted to a summary line. Exit code `1` on any CRITICAL finding, safe for CI gating.

See [Integrity Checks](integrity.md) for what each check detects and why it matters.

---

## `calibra audit` — full diagnostic report

```bash
calibra /data/robot_demos.h5
calibra lerobot/pusht --policy diffusion
calibra /data/demo.h5 --policy act --json
calibra /data/robot_demos.h5 --html-out report.html   # save visual HTML dashboard
calibra /data/demos.h5 --cache-dir .calibra/cache     # incremental analysis
```

Runs four analyzers over every episode and flags anomalies with bootstrap confidence intervals and per-episode outlier detection. The `--html-out` dashboard includes a **Dataset Health Score panel** — a composite 0–100 score derived from diagnostic flags, broken down into four sub-scores: Quality, Synchrony, Coverage, and Integrity (color-coded green/yellow/red).

`--cache-dir DIR` enables incremental analysis: the pipeline result is stored in a file-based cache keyed by a SHA-256 fingerprint of the episode manifest. On unchanged data, the next run returns instantly from cache. Useful when collecting daily demos and re-auditing the same dataset repeatedly.

---

## `calibra compare` — evidence-backed cross-dataset comparison

```bash
calibra compare /data/my_demos pusht
calibra compare hf://lerobot/my_dataset aloha
calibra compare /data/robot.h5 aloha --format hdf5 --gripper-dims 6,13
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra compare — my_dataset  vs.  aloha
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reference: lerobot/aloha_mobile_cabinet  (position-command · 14D · 85 episodes)
Yours:     my_dataset  (120 episodes)

────────────────────────────────────────────────────────
VELOCITY DISCONTINUITY RATE
  Yours:  12.1%
  aloha   1.3%
  Delta:  +10.8%  ▲

  Significantly rougher than aloha_mobile_cabinet.
  If using position commands: investigate control noise or
  abrupt operator corrections.

  Confidence: HIGH · [HIGH · n=2 (aloha_sim, aloha_mobile)]
────────────────────────────────────────────────────────
JERK SPIKE RATE
  Yours:  8.4%
  aloha   0.7%
  Delta:  +7.7%  ▲

  Higher spike rate than reference. Check for dropped
  frames, bad episode boundaries, or bimodal speed profiles.

  Confidence: MODERATE · [LOW-MODERATE · n=1 (aloha_sim)]
────────────────────────────────────────────────────────

RECOMMENDED ACTIONS
────────────────────────────────────────────────────────
  Prune episode(s) 14, 22, 41 — jerk outliers detected by MAD analysis.
  Velocity discontinuity rate is 12.1% (above 4% position-control
  threshold). Investigate command packet drops, hardware communication
  lag, or abrupt operator corrections.
────────────────────────────────────────────────────────
```

Every interpretation is backed by a falsifiable claim in `calibra/claims/` with an evidence count, confidence rating, and a stated falsification condition.

---

## `calibra certify` — structured pass/fail certification

```bash
calibra certify /data/my_demos
calibra certify /data/my_demos --reference aloha --policy diffusion --strict
calibra certify hf://lerobot/my_dataset --json   # for CI pipelines
calibra certify /data/my_demos --report results/my_demos/latest.json  # write CalibraReport JSON
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA CERTIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset  : my_demos
  Episodes : 120
  Steps    : 180000
  Policy   : diffusion
  Reference: aloha

  ──────────────────────────────────────────────────────────
  ⚠  PROVISIONALLY CERTIFIED

  Warnings:
    • ldlj: Mean LDLJ = -12.4 (threshold: >-10). Action trajectories
      contain significant jerk.

  ──────────────────────────────────────────────────────────
  REMEDIATION CHECKLIST
  ──────────────────────────────────────────────────────────
  1. [WARNING] ldlj: High jerk in demonstration data forces the policy
     to learn discontinuous action transitions. Consider applying action
     smoothing (e.g. Savitzky-Golay) before training.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Exit codes: `0` = CERTIFIED, `1` = PROVISIONALLY CERTIFIED (warnings), `2` = NOT CERTIFIED (critical failures). Wire into CI with `--json` for machine-readable output.

`--report PATH` writes a schema-versioned **CalibraReport** JSON consumed by `calibra site`.

---

## `calibra prune` — coreset selection

```bash
calibra prune /data/100k_episodes --keep 0.3 --out coreset.json
calibra prune /data/my_ds --keep 0.5 --quality-only
calibra prune /data/my_ds --keep 0.25 --max-spike-rate 0.03 --max-vel-disc-rate 0.08

# Write a schema-versioned CalibraReport with per-episode verdicts (recommended)
calibra prune /data/demos.h5 --keep 0.3 --report results/my_ds/latest.json

# GR00T fine-tuning: strict quality thresholds + entropy-weighted diversity
calibra prune demos.hdf5 --keep 0.3 --policy gr00t --report results/franka/latest.json

# Incremental analysis: skip re-running the pipeline on unchanged episodes
calibra prune /data/demos.h5 --keep 0.3 --cache-dir .calibra/cache --report results/latest.json
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA PRUNING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Original episodes  : 1000
  Quality failures   : 87   (removed in Stage 1)
  Diversity pruned   : 613  (removed in Stage 2)
  Coreset size       : 300  (30.0% of original)
  Method             : quality_filter + greedy_max_coverage
────────────────────────────────────────────────────────
  To use: filter your dataset to the episode IDs in keep_episode_ids.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Two-stage pipeline:
- **Stage 1 — Quality filter:** removes episodes that fail kinematic/temporal thresholds (jerk spike rate, velocity discontinuity, dropout, LDLJ, minimum length).
- **Stage 2 — Greedy max-coverage:** from the quality-passing pool, selects the K most behaviorally diverse episodes using farthest-point sampling on action-space statistics. O(N × K) — handles ~50k episodes without approximation.

Use `--entropy-weight 0.4` (or `--policy gr00t`) to bias selection toward high-entropy (informationally rich) episodes. Use `--strategy influence` to select episodes based on estimated learning value (combining action novelty, task contact representation, and Shannon entropy).

`--report PATH` writes a schema-versioned **CalibraReport JSON** with `episode_verdicts` — approved/rejected episode IDs, per-episode reason codes (e.g. `jerk_spike`, `diversity_pruned`), quality scores, and SHA-256 content hashes.

`--cache-dir DIR` caches the diagnostic pipeline result keyed by episode manifest fingerprint. On repeated runs with unchanged data, skips the pipeline — typically 10–50× faster on large datasets collected incrementally.

---

## `calibra corrupt` — validate metric sensitivity

```bash
calibra corrupt lerobot/pusht --drop-frames 0.10
calibra corrupt /data/robot.h5 --inject-spikes 0.05
calibra corrupt lerobot/pusht --add-jitter-ms 50 --drop-frames 0.08
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
calibra corrupt — pusht
Corruptions: drop_frames=10.0%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Metric                      Original   Corrupted       Δ  React
──────────────────────────────────────────────────────────────────
  Timestamp dropout rate         0.0%       9.4%    +9.4%  🔴
  Timestamp jitter CV          3.0e-06    8.1e-06  +5.1e-06 🟡
  Jerk spike rate                4.9%       5.2%    +0.3%   —
  Velocity discontinuity        16.7%      16.9%    +0.2%   —
```

Inject synthetic corruptions into a known-good dataset to verify that your metrics actually respond to the defects they claim to detect.

---

## `calibra retarget` — convert absolute EEF actions to relative deltas

```bash
calibra retarget /data/isaac_lab_demos.h5 --out /data/retargeted/
calibra retarget /data/demos.h5 --pad --out retargeted/
calibra retarget /data/demos.h5 --obs-key-pos robot0_eef_pos \
                                 --obs-key-quat robot0_eef_quat
```

NVIDIA GR00T N1.7+ uses a **Relative End-Effector (EEF)** action space. Isaac Lab and robomimic HDF5 datasets record actions in absolute world-frame coordinates. `retarget` converts absolute 7-DoF poses `[x, y, z, qx, qy, qz, qw]` into 6-DoF local-frame deltas `[dx, dy, dz, droll, dpitch, dyaw]`.

Use `--pad` to append a zero row so output shape is `(T, 6)` instead of `(T−1, 6)` when your policy requires fixed-length sequences.

---

## `calibra predict` — predict training outcome before spending GPU time

```bash
calibra predict /data/my_demos.h5
calibra predict lerobot/my_dataset --policy diffusion --reference aloha
calibra predict /data/my_demos.h5 --policy gr00t --record-outcome 0.82
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA TRAINING OUTCOME PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dataset  : my_demos  ·  Episodes: 120  ·  Policy: gr00t
  🟢  Predicted Success: 81%  [range 71%–91%]  —  GOOD
  ──────────────────────────────────────────────────────────
  ⚠️  -8.0pt  ldlj
     Mean LDLJ = -12.4. High jerk forces discontinuous action transitions.
  ──────────────────────────────────────────────────────────
  NEXT STEPS
  ✓ Data quality is sufficient. Proceed with training.
  After training, close the loop:
    calibra predict <dataset> --record-outcome <actual_success_rate>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`--record-outcome RATE` stores the observed training success rate alongside the diagnostic fingerprint in `~/.calibra/outcomes.jsonl`. Future predictions on similar datasets blend the heuristic score with these empirical observations via inverse-distance weighting.

---

## `calibra card` — HuggingFace dataset quality card

```bash
calibra card /data/my_demos.h5
calibra card lerobot/my_dataset --policy diffusion --out quality_card.md
calibra card /data/my_demos.h5 --push   # push directly to HuggingFace Hub README
```

Generates a structured Markdown quality card with certification badge, per-metric status table, and predicted training outcome. Embed it in your dataset's HuggingFace Hub README so other researchers can see data quality at a glance.

---

## `calibra watch` — real-time teleoperation quality monitor

```bash
calibra watch /data/collection_session/
calibra watch /data/session/ --remediate          # print fix instructions on failure
calibra watch /data/session/ --log-file session.jsonl

# Stream mode: pipe metrics from your collection script
python collect_demos.py | calibra watch --stream --remediate
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA WATCH — real-time data quality monitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Remediation advice: ON
  Watching: /data/collection_session/

  ✅ [   1] ep_001.h5         PASS — all metrics OK
  ✅ [   2] ep_002.h5         PASS — all metrics OK
  ❌ [   3] ep_003.h5         FAIL — jerk_spike_rate = 0.087
       ↳ RE-RECORD: Move more smoothly — avoid abrupt stops and direction changes.
  ✅ [   4] ep_004.h5         PASS — all metrics OK
```

`--remediate` prints a specific operator instruction on every FAIL/WARN. `--stream` reads JSON metric lines from stdin for integration with teleoperation software without filesystem round-trips.

---

## `calibra score` — composite 0–100 quality score

```bash
calibra score /data/robot_demos.h5
calibra score lerobot/my_dataset --policy diffusion
calibra score /data/my_ds --reference aloha --json
calibra score hf://lerobot/pusht_image --badge   # print markdown badge for dataset cards
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dataset  : my_demos
  Episodes : 120  ·  Steps: 180000

────────────────────────────────────────────────────────────
  🟢  78.0 / 100  —  Good
────────────────────────────────────────────────────────────

  Temporal Stability       22.00/25  [█████████████████░░░]  88%
  Control Smoothness       26.00/35  [██████████████░░░░░░]  74%
  Coverage / Diversity     19.00/25  [███████████████░░░░░]  76%
  Task Structure           11.00/15  [██████████████░░░░░░]  73%

  0 critical flags  ·  3 warnings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Score categories: 90–100 Excellent, 75–89 Good, 60–74 Fair, 40–59 Poor, 0–39 Critical. Exit codes: `0` = Good or better (≥75), `1` = Fair or Poor (40–74), `2` = Critical (<40).

---

## `calibra sim2real` — sim-to-real distribution gap

```bash
calibra sim2real /data/sim_demos.h5 /data/real_demos.h5
calibra sim2real lerobot/sim_dataset /data/real.h5 --policy pi0
calibra sim2real /data/sim.h5 /data/real.h5 --json
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CALIBRA SIM-TO-REAL GAP ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Sim dataset  : isaac_lab_pick  (500 eps)
  Real dataset : real_pick       (120 eps)

  🟡  Overall Transfer Risk: MEDIUM
  📊  Pre-training Alignment Index (PAI): 71.3%

  🟢 Ldlj Gap                  [LOW]   Sim: -6.2   Real: -8.1   Δ = 1.9
  🟡 Action Kl Divergence      [MEDIUM] Value: 0.73
  🟢 Sim Coverage Of Real      [LOW]   Value: 0.81
  🟢 Control Frequency Gap     [LOW]   Sim: 50Hz   Real: 50Hz   Δ = 0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Reports overall transfer risk (LOW / MEDIUM / HIGH / CRITICAL) and a Pre-training Alignment Index (PAI, 0–100%). Exit codes: `0` = LOW or MEDIUM, `1` = HIGH, `2` = CRITICAL.

---

## `calibra transfer` — cross-embodiment compatibility

```bash
calibra transfer /data/source_robot.h5 /data/target_robot.h5
calibra transfer lerobot/aloha_mobile_cabinet lerobot/svla_so100_pickplace
```

Scores reuse compatibility across action dimensionality, control frequency, trajectory smoothness, episode length, and action range overlap. Levels: DIRECT (mix freely), ADAPT (normalise or retarget first), DIFFICULT, INCOMPATIBLE. Exit codes: `0` = DIRECT or ADAPT, `1` = DIFFICULT, `2` = INCOMPATIBLE.

---

## `calibra cure` — automatic data remediation

```bash
calibra cure /data/robot_demos.h5 --out cured/
calibra cure /data/demos.h5 --remedy smooth,trim --out cured/
calibra cure lerobot/pusht --hz 10 --out cured/ --format lerobot
```

Applies kinematic and temporal fixes to every episode and writes cleaned per-episode `.npz` files. Default remedy pipeline: `smooth,interpolate,trim` — Savitzky-Golay filtering, uniform resampling, and dead-time trimming. Use `--remedy` to apply a subset, `--hz` to pin the output control frequency. Writes a `cure_manifest.json` with original and cured step counts.

---

## `calibra audit-all` — bulk dataset auditor

```bash
calibra audit-all --org lerobot                        # audit every dataset in an HF org
calibra audit-all --org lerobot --out ./results --workers 8
calibra audit-all --dataset lerobot/pusht lerobot/aloha_sim_insertion_human
calibra audit-all --org lerobot --force                # re-audit even if cached
calibra audit-all --org lerobot --limit 5 --dry-run    # preview without running
```

```
Discovering datasets (org=lerobot) ...
Found 47 dataset(s).
[1/47] lerobot/pusht  auditing ...
[1/47] lerobot/pusht  OK  score=74.2 grade=C cert=provisional  8.3s
...
Done.  audited=45  skipped=2  failed=0  mean_score=71.8
Manifest: results/manifest.json
```

Bulk-audits a HuggingFace org or explicit dataset list in parallel. Writes CalibraReport JSONs to:

```
results/<org>/<slug>/<revision-sha[:8]>/<timestamp>.json
results/<org>/<slug>/latest.json    ← always up-to-date symlink
```

Skips datasets whose current revision is already cached; use `--force` to re-audit. Requires `pip install huggingface-hub`.

---

## `calibra site` — static leaderboard website

```bash
calibra site --results ./results --out ./site
calibra site --results ./results --out ./site --title "My Robot Lab Leaderboard"
```

Reads the `results/` directory tree produced by `audit-all` and generates a self-contained static website:

| Output | Description |
|---|---|
| `site/index.html` | Sortable, filterable dataset leaderboard |
| `site/<org>/<slug>/index.html` | Per-dataset detail page |
| `site/<org>/<slug>/badge.svg` | Embeddable quality badge |
| `site/<org>/<slug>/history.json` | Score history across dataset revisions |

No build step, no dependencies — host on GitHub Pages, Netlify, or any static file server.

---

## `calibra serve` — local REST API server and web dashboard

```bash
calibra serve                    # start on localhost:7842
calibra serve --port 8000
calibra serve --host 0.0.0.0
```

Starts a local HTTP server exposing all Calibra diagnostics as a REST API and serving the visual web dashboard at `http://localhost:7842`. Use `--host 0.0.0.0` to expose on all network interfaces.

---

## `calibra benchmark` — full vs. random vs. Calibra comparison

```bash
# Single retention level, purely simulated
calibra benchmark lerobot/pusht --keep 0.3 --policy diffusion

# Full retention curve in one shot
calibra benchmark lerobot/pusht --sweep

# Custom retention levels
calibra benchmark lerobot/pusht --sweep --fractions 0.10,0.30,0.50,1.00

# Substitute real measured results wherever they've been recorded
calibra benchmark lerobot/pusht --sweep --experiment-id partner-a-pusht

calibra benchmark lerobot/pusht --keep 0.3 --json   # machine-readable
```

Runs diagnostics + the heuristic outcome predictor on three conditions — the raw dataset, a randomly pruned subset, and the Calibra coreset — and reports GPU-hours and predicted success rate for each. GPU-hours are simulated by default: linear scaling of `--base-gpu-hours` (default 24.0) by episode-count fraction.

| Flag | Description |
|---|---|
| `--keep FRACTION` | Retention fraction for a single-point comparison (default 0.3). Ignored with `--sweep`. |
| `--sweep` | Run the full design-partner retention curve instead of one `--keep` value: full baseline, then random vs. Calibra at every level in `--fractions`. |
| `--fractions LIST` | Comma-separated retention fractions for `--sweep` (default `0.10,0.25,0.50,0.75,1.00`, matching the design-partner protocol). |
| `--experiment-id ID` | Substitute real measured GPU-hours / eval success rate from `calibra experiment record` wherever a matching condition and retention level has been logged. Falls back to simulated values for anything not yet measured. |
| `--base-gpu-hours H` | GPU-hours to train on the full (100%) dataset, used for simulated scaling (default 24.0). |
| `--policy` | Policy family for the outcome predictor (`bc-mlp`, `act`, `diffusion`, `gr00t`, ...). |
| `--json` | Machine-readable output. |

**Simulated vs. measured.** Every reported number is tagged `(simulated)` or `(measured)`, and the report carries an overall status:

- **SIMULATED** — nothing measured yet; the numbers are predictions from the heuristic outcome model and linear GPU-hour scaling. Not a case study.
- **PARTIAL MEASUREMENT** — some conditions are measured, others still simulated. Not safe to present as a validated result — mixing real and predicted numbers without labeling them is misleading.
- **CASE STUDY / VALIDATED** — full, random, and Calibra are all backed by real recorded training runs at that retention level. Safe to report as a validated case study.

Compute savings are computed from GPU-hours, not episode-count reduction — once real numbers are mixed in, the two can diverge (dataloader/I/O overhead doesn't shrink proportionally with data).

---

## `calibra experiment` — record and report measured training results

```bash
# Log one training run's result
calibra experiment record --experiment-id partner-a-pusht \
    --dataset partner-a/pusht_v3 --condition calibra --retention 25 \
    --n-episodes 300 --policy act --eval-success-rate 0.84 \
    --gpu-hours 19.8 --seed 0

calibra experiment list --experiment-id partner-a-pusht
calibra experiment report --experiment-id partner-a-pusht
calibra experiment report --experiment-id partner-a-pusht --json
```

Records the *results* of real training runs — GPU-hours, wall-clock time, energy, eval success rate — against the design-partner protocol's three conditions (`full`, `random`, `calibra`) at a given retention percentage. This command doesn't run training itself; training happens in whatever pipeline the partner already uses (`lerobot-train`, a custom loop, etc.) — `calibra experiment` just logs what came out of it, consistently, so it can be compared and fed into `calibra benchmark --experiment-id`.

Stored as JSON Lines at `~/.calibra/experiments.jsonl` by default (override with `--path`). Local only — never synced to any network endpoint, matching the rest of Calibra's on-prem posture.

`calibra experiment report` prints the full retention curve for one experiment, the Calibra-vs-random delta at each level, and which `(retention%, condition)` pairs the protocol still expects but haven't been recorded (`10/25/50/75/100%` × `full/random/calibra`, minus the combinations that don't apply — `full` only at 100%, `random`/`calibra` never at 100%).

### Partner workflow

```text
1. Run Calibra           calibra prune / calibra benchmark --sweep
2. Train Full / Random / Calibra   (partner's own training pipeline)
3. Record measured results         calibra experiment record --condition ... --retention ...
4. Run benchmark with experiment ID   calibra benchmark --sweep --experiment-id <ID>
5. Generate comparison              calibra experiment report --experiment-id <ID>
```

Repeat steps 2–3 for each condition/retention level until `calibra experiment report` shows the protocol complete, at which point `calibra benchmark --sweep --experiment-id <ID>` reports `CASE STUDY / VALIDATED`.
