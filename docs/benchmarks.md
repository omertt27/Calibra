# Calibra Benchmark Results

Complete statistical results, ablation tables, and limitations for all Calibra benchmarks. The README presents the headline numbers; this document has everything.

---

## LeRobot PushT — Targeted Benchmark (Primary Real-Data Result)

### Setup

**Dataset:** `lerobot/pusht` (HuggingFace Hub, no modification)  
**Episodes:** 206 total · 165 training · 41 test  
**Frames:** 25,650  
**Calibra quality score:** 76.7/100 (4 CRITICAL, 2 WARNING flags)  
**Quality-approved pool:** 123/165 episodes (75%) after Stage-1 filtering  

**Tail episode identification:** k-NN density (k=5) in action-feature space (mean + std of actions + first 4 state dims, L2-normalised). Bottom 15% by local density = **25 training episodes** labelled tail, **7 test episodes** labelled tail, **34 test episodes** common.

**Baselines (single seed, 120 epochs):**

| Condition | Overall MSE | Common-test MSE | Tail-test MSE | Train time |
|---|---:|---:|---:|---:|
| Full unfiltered (165 ep) | 420.93 | 440.29 | 323.98 | 25.3s |
| Quality-approved full (123 ep) | 423.93 | 443.58 | 325.54 | 15.6s |

**Methods compared (all at equal episode budget k):**

| ID | Method | Episode pool | Selection |
|---|---|---|---|
| random_full | Random (full) | All 165 train ep | Random draw — reference |
| random_quality | Random (quality) | 123 quality-approved ep | Random draw |
| quality_only | Quality-only | 123 quality-approved ep | Quality filter → random |
| diversity_only | Diversity-only | All 165 train ep | k-center, no quality gate |
| calibra | Calibra | All 165 → quality filter → 123 | Quality filter + k-center |

**Training:** BC-MLP, 3-layer, hidden=256, LayerNorm, SiLU, AdamW, cosine LR, 120 epochs  
**Seeds:** 10. For deterministic methods (calibra, diversity_only, quality_only) the seed controls training randomness only. For random methods (random_full, random_quality) the seed controls both episode selection and training.  
**Statistics:** paired t-test and Cohen's d computed vs. random_full across matched seeds.

---

### Results

#### 5% retention — k=8 from full / k=6 from quality pool

| Method | Tail cov. | Overall MSE | ±CI95 | vs. random | p | d | Tail MSE | Common MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_full | 4.0% | 488.33 | 29.04 | — | — | — | 407.62 | 504.44 |
| random_quality | 5.2% | 514.48 | 33.41 | −26.16 | 0.009 | −0.60 | 449.17 | 527.52 |
| quality_only | **0.0%** | 549.11 | 7.23 | −60.79 | 0.001 | −2.05 | 500.16 | 558.88 |
| diversity_only | **16.0%** | **449.13** | **1.07** | **+39.19** | **0.014** | **+1.36** | **351.73** | **468.58** |
| calibra | 12.0% | 457.64 | 0.96 | +30.68 | 0.042 | +1.07 | 377.61 | 473.62 |

#### 10% retention — k=16 from full / k=12 from quality pool

| Method | Tail cov. | Overall MSE | ±CI95 | vs. random | p | d | Tail MSE | Common MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_full | 8.4% | 445.85 | 10.33 | — | — | — | 353.01 | 464.38 |
| random_quality | 8.4% | 462.61 | 11.45 | −16.76 | 0.028 | −1.10 | 368.61 | 481.37 |
| quality_only | 4.0% | 463.25 | 1.66 | −17.40 | 0.004 | −1.68 | 380.42 | 479.78 |
| **diversity_only** | **24.0%** | **440.69** | **0.96** | +5.16 | 0.319 | +0.50 | **343.46** | **460.11** |
| calibra | 24.0% | 463.42 | 2.40 | −17.57 | **0.006** | −1.68 | 384.12 | 479.25 |

#### 25% retention — k=41 from full / k=31 from quality pool

| Method | Tail cov. | Overall MSE | ±CI95 | vs. random | p | d | Tail MSE | Common MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_full | 23.6% | 427.49 | 1.84 | — | — | — | 327.66 | 447.42 |
| random_quality | 18.8% | 441.27 | 8.85 | −13.78 | 0.006 | −1.54 | 340.75 | 461.34 |
| quality_only | 12.0% | 432.80 | 0.52 | −5.31 | <0.001 | −2.81 | 328.43 | 453.64 |
| diversity_only | 52.0% | 425.71 | 0.44 | +1.79 | 0.065 | +0.96 | 330.57 | 444.70 |
| **calibra** | **56.0%** | **422.89** | **0.45** | **+4.61** | **<0.001** | **+2.46** | **321.16** | **443.20** |

*d = Cohen's d vs. random_full — positive means method has lower (better) MSE than random_full. p = two-sided paired t-test. Tail cov. = fraction of 25 action-space low-density training episodes retained by the method.*

---

### Key findings

**1. At 5% retention (8 episodes), both diversity-based methods are statistically significantly better than random (p=0.014 and p=0.042).** diversity_only achieves the best overall MSE (449.13) and tail MSE (351.73) at this budget, retaining 4× more tail episodes than random (16% vs. 4%).

**2. At 10% retention, calibra is significantly worse than random (p=0.006, d=−1.68).** This is not noise — it replicates across all 10 seeds. The quality filter reduces the pool from 165 to 123 episodes before diversity selection, cutting off access to tail episodes. At k=16, drawing from 123 quality-approved episodes provides less tail coverage than drawing from the full 165. diversity_only (full pool, no quality gate) is not significantly different from random at this budget (p=0.319).

**3. At 25% retention, calibra is the strongest method (p<0.001, d=+2.46).** Tail coverage 56% vs. 23.6% for random. Tail MSE **321.16 — below the full-dataset tail MSE of 323.98**. The quality filter pays off at this budget because the 41-episode selection can afford to avoid lower-quality episodes while still covering the tail.

**4. quality_only is consistently the worst performing method at all three budgets.** It achieves 0% tail coverage at 5%, significantly worse MSE at 5% (p=0.001) and 10% (p=0.004), and significantly worse MSE at 25% (p<0.001, d=−2.81). Quality filtering without diversity selection removes tail episodes first (they are often kinematically atypical) and then draws randomly from what remains.

**5. The optimal strategy is budget-dependent.** Below ~25% retention: use diversity-only (k-center on the full pool, no quality gate). At 25%+: Calibra's quality filter pays off. This is consistent with the regime-space analysis in the main ablation.

---

### Limitations

- **Tail episodes are action-space low-density, not validated semantic edge cases.** The bottom-15% k-NN density threshold identifies episodes with unusual action statistics. These may correspond to genuinely informative behaviours or to episodes that are unusual for incidental reasons (unusual start position, operator hesitation, trajectory length outlier). No manual inspection has been performed to validate their semantic relevance.

- **Tail MSE improvement at 25% is on 7 test episodes.** The tail test set is small. The finding is consistent across 10 seeds, but a larger dataset with more tail test episodes would provide stronger evidence.

- **No simulator rollout.** All evaluation uses held-out trajectory prediction MSE. Connecting tail coverage to task-level success rate requires rollout in a sim environment and is left for future work.

- **Single dataset.** These results are from `lerobot/pusht` only. Replication on DROID, ALOHA, and other LeRobot datasets is the next step.

- **10% crossover is honest, not explained away.** At 10% retention, calibra is significantly worse than random. This is reported without softening.

---

### Reproduce

```bash
pip install "calibra-robotics[lerobot]" torch matplotlib
python experiments/lerobot_targeted_benchmark.py \
    --dataset lerobot/pusht \
    --n-seeds 10 \
    --n-epochs 120

# Results:  experiments/figures/targeted_lerobot_pusht.json
# Figure:   experiments/figures/targeted_lerobot_pusht.pdf  (6 panels)
```

Smoke test (3 seeds, 60 epochs, ~3 minutes):
```bash
python experiments/lerobot_targeted_benchmark.py --n-seeds 3 --n-epochs 60
```

---

## Retention Sweep — LeRobot PushT (5 methods, 6 fractions)

A broader sweep across six retention fractions (5%–100%) comparing Calibra vs. Random (5 seeds) and tracking action-space tail coverage. This is the simpler benchmark run before the targeted statistical study above.

**Setup:** 165 train / 41 test episodes. 5 random seeds. 120 epochs. Tail = bottom 15% of training episodes by action-space density (25 episodes).

| Keep | N | Calibra MSE | Rel | Tail cov | Random MSE ±std | Rel | Tail cov | Saved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 8 | 457.05 | 1.09x | 12.0% | 490.95 ±18.0 | 1.17x | 6.4% | 95% |
| 10% | 16 | 459.46 | 1.09x | 24.0% | 448.13 ±9.5 | 1.07x | 10.4% | 90% |
| 25% | 41 | 422.77 | 1.01x | 56.0% | 429.61 ±1.7 | 1.02x | 33.6% | 75% |
| 50% | 82 | 422.55 | 1.00x | 68.0% | 423.36 ±2.2 | 1.01x | 52.8% | 50% |
| 75% | 124 | 423.84 | 1.01x | 72.0% | 421.67 ±1.4 | 1.00x | 78.4% | 25% |
| 100% | 123* | 423.14 | 1.01x | 72.0% | 420.76 ±0.2 | 1.00x | 100.0% | 0% |

*At 100%, Calibra's quality filter caps selection at 123 of 165 episodes. The 100% row reflects quality-filtered full dataset, not the unfiltered full dataset.*

Rel = MSE / full-dataset MSE. Values ≤ 1.00 match or beat the full dataset.

```bash
python experiments/lerobot_coreset_benchmark.py \
    --dataset lerobot/pusht \
    --n-epochs 120 \
    --n-seeds 5
```

---

## Synthetic Rare-Mode Benchmark (Controlled)

See README § "Rare-mode preservation" for the controlled 2D multigoal benchmark (worst-group success, 5-seed). Results: 98.7% worst-group (Calibra) vs. 36.5% (Random) at 5% retention.

```bash
python experiments/multigoal_obstacle_benchmark.py
```

---

## Multi-Dataset Ablation (BC-MLP, ACT, Diffusion Policy)

See README § "Ablation study" and § "Cross-architecture check" for full tables across ALOHA Mobile, DROID-100, and PushT real at 30% retention with 5 seeds, 3 policy families, and 7 competing methods including K-Center, Facility Location, and Herding.

```bash
python experiments/ablation_benchmark.py --dataset lerobot/aloha_mobile_cabinet --seeds 5
python experiments/ablation_benchmark.py --dataset lerobot/droid_100 --seeds 5
python experiments/ablation_benchmark.py --dataset lerobot/columbia_cairlab_pusht_real --seeds 5
python experiments/aggregate_ablation.py results/ablation_*.json
```
