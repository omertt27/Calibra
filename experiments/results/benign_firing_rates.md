# Calibra Detector Calibration Benchmark

Measured by `experiments/benign_firing_rate_benchmark.py`.

## Two-Axis Calibration Table

| Detector | Clean flag rate | Detection rate | Signal ratio | Concentration |
|---|---|---|---|---|
| jitter_cv | 3.4% | 37.1% | 11× | 62% in top 5 eps |
| dropout_rate | 0.0% | 18.8% | — | 62% in top 5 eps |
| spike_rate | 4.7% | 51.2% | 11× | 62% in top 5 eps |
| vel_disc_rate | 1.9% | 30.5% | 16× | 62% in top 5 eps |
| ldlj | 0.0% | 50.0% | — | 62% in top 5 eps |

## Per-Dataset Results

### lerobot/pusht (task: pusht, n=206)

Episode flag rate: 24/206 = 11.7%
Top-5 episode concentration: 24% of flags

| Detector | Flagged | Rate | 95% CI |
|---|---|---|---|
| jitter_cv | 14/206 | 6.8% | (4.1%, 11.1%) |
| dropout_rate | 0/206 | 0.0% | (0.0%, 1.8%) |
| spike_rate | 3/206 | 1.5% | (0.5%, 4.2%) |
| vel_disc_rate | 8/206 | 3.9% | (2.0%, 7.5%) |
| ldlj | 0/206 | 0.0% | (0.0%, 1.8%) |

**Corruption detection** (corrupt fraction: 15%)

| Detector | Detected / Corrupted | Episode detection rate† |
|---|---|---|
| jitter_cv | 18/30 | 60.0% |
| dropout_rate | 7/30 | 23.3% |
| spike_rate | 5/30 | 16.7% |
| vel_disc_rate | 14/30 | 46.7% |
| ldlj | 0/30 | 0.0% |

### lerobot/aloha_sim_insertion_scripted (task: aloha, n=50)

Episode flag rate: 4/50 = 8.0%
Top-5 episode concentration: 100% of flags

| Detector | Flagged | Rate | 95% CI |
|---|---|---|---|
| jitter_cv | 0/50 | 0.0% | (0.0%, 7.1%) |
| dropout_rate | 0/50 | 0.0% | (0.0%, 7.1%) |
| spike_rate | 4/50 | 8.0% | (3.2%, 18.8%) |
| vel_disc_rate | 0/50 | 0.0% | (0.0%, 7.1%) |
| ldlj | 0/50 | 0.0% | (0.0%, 7.1%) |

**Corruption detection** (corrupt fraction: 15%)

| Detector | Detected / Corrupted | Episode detection rate† |
|---|---|---|
| jitter_cv | 1/7 | 14.3% |
| dropout_rate | 1/7 | 14.3% |
| spike_rate | 6/7 | 85.7% |
| vel_disc_rate | 1/7 | 14.3% |
| ldlj | 7/7 | 100.0% |

## Interpretation Note

These rates use within-dataset MAD-based outlier detection.
Even clean datasets have episodes at the tail of their own quality
distribution — the benign firing rate captures how often those tails
exceed the MAD threshold in practice.

**A detected anomaly is not the same as confirmed corruption.**
Review flagged episodes using `calibra review` before deciding to drop,
downweight, or annotate them.

† Episode detection rate is episode-level: an episode is counted as detected
if the detector fired anywhere in it. It does not distinguish whether the
detector fired at the corrupted region or elsewhere in the episode.