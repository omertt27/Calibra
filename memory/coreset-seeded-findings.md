---
name: coreset-seeded-findings
description: Seeded 3-dataset coreset ablation results and the paper-framing decision they support
metadata:
  type: project
---

Seeded (5-seed, paired-t) coreset ablation on ALOHA / DROID / PushT-real (keep 30%, BC-MLP, RTX 2080), run 2026-07-07/08. Scripts: `experiments/ablation_benchmark.py` (seeded, per-condition `per_seed_mse`) + `experiments/aggregate_ablation.py` (mean rank / W-T-L / oracle headroom). Baselines K-Center/Herding/Facility added on the same behavioral features.

Key results (mean vs-random, mean rank):
- **Diversity-only +29.5%, rank 2.00** — best/tied-best, most consistent, within 1.3pt of oracle (+30.8%). This IS Calibra's diversity stage.
- K-Center greedy +24.0%, rank 2.00 (higher variance: rank-4 on DROID).
- Calibra full (quality+diversity) +24.5%, rank 2.67 — **significantly loses to diversity-only on 2/3** (DROID p=0.003, PushT p=0.001).
- Herding is bad (negative).

Decisions this locks in:
1. **Single-seed runs were misleading** — earlier "quality filter is negative on DROID (−2.4%)" and "K-Center beats Calibra on 2/3" were training-noise artifacts (±10% rel). Always seed + paired-t.
2. **Control-mode (position vs velocity) is NOT the cause** of DROID's quality-filter underperformance — velocity-mode metrics ≈ position within noise. Don't build the per-control-mode fix for this.
3. **Oracle headroom is tiny (+1.3pt)** → do NOT claim adaptive regime-aware selection. See [[paper-framing-decision]].
4. **Actionable: change Calibra's default** from quality+diversity to **diversity-only**, and gate the quality filter behind *detected corruption* (where it helps: synthetic / PushT-injected / L4 in RESULTS.md). Worth ~+5pt on clean data.

Paper framing supported: "observability platform whose coverage selection matches the best published coreset (K-Center) and is more consistent; quality-filtering is corruption-gated" — NOT "best coreset algorithm" and NOT "adaptive selection".
