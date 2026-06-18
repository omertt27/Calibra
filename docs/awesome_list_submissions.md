# Awesome List Submission Guide

This document contains ready-to-paste entry text and PR guidance for
submitting Calibra to the three most relevant awesome lists.

---

## 1. [awesome-robot-learning](https://github.com/makezur/awesome-robot-learning)

**Target section:** `Data & Datasets` or `Tooling / Infrastructure`

**Entry text:**
```markdown
- [Calibra](https://github.com/omerTT/Calibra) — Dataset observability and coreset selection for robot imitation learning. Detects jerk spikes, velocity discontinuities, and redundancy in demonstration datasets before policy training. Includes a cross-dataset analysis of 15 public LeRobot v2 datasets. `pip install calibra-robotics`
```

**PR title:** `Add Calibra: dataset observability toolkit for imitation learning`

**PR body template:**
```
Hi! I'd like to add [Calibra](https://github.com/omerTT/Calibra) to the
tooling/data section.

Calibra is an open-source dataset observability toolkit for robot imitation
learning. It runs four deterministic estimators (jerk spike rate, velocity
discontinuity rate, timing jitter, LDLJ) over any LeRobot v2 dataset and
reports per-episode anomalies with bootstrap confidence intervals.

**Why it belongs here:**
- Profiled 15 public datasets (ALOHA, DROID, BridgeData V2, PushT, SVLA)
- Found 25× spike-rate difference between human vs scripted planners
- Available on PyPI: `pip install calibra-robotics`
- MIT licence, CI/CD, fully documented

Technical report available at: [docs/report/](https://github.com/omerTT/Calibra/tree/main/docs/report)
```

---

## 2. [awesome-imitation-learning](https://github.com/kristery/Awesome-Imitation-Learning)

**Target section:** `Tools` or `Environments / Datasets`

**Entry text:**
```markdown
- [Calibra](https://github.com/omerTT/Calibra) - Open-source dataset quality auditing for imitation learning. Flags jerk spikes, velocity discontinuities, timing artifacts, and redundant episodes across any LeRobot v2 dataset. Includes a coreset pruner that reduces dataset size by up to 70% while preserving behavioural diversity.
```

**PR title:** `Add Calibra dataset observability tool`

---

## 3. [awesome-lerobot](https://github.com/radekosmulski/awesome-lerobot)

**Target section:** `Dataset Tools` or `Utilities`

**Entry text:**
```markdown
- [Calibra](https://github.com/omerTT/Calibra) — Dataset observability for LeRobot v2 datasets. Run `calibra compare hf://lerobot/my_dataset aloha` to compare your dataset against validated reference profiles and detect quality issues before training.
```

**PR title:** `Add Calibra: dataset quality auditing for LeRobot v2`

**PR body template:**
```
Calibra is a companion tool for the LeRobot ecosystem.

It loads any LeRobot v2 dataset via the standard HuggingFace adapter and
computes four kinematic quality metrics per episode:
- Jerk spike rate
- Velocity discontinuity rate
- Timing jitter CV
- Log dimensionless jerk (LDLJ)

It ships with 15 pre-computed reference profiles (ALOHA family, DROID,
BridgeData V2, PushT, SVLA SO-100) so you can compare your dataset against
known-good baselines.

Install: `pip install calibra-robotics`
Repo: https://github.com/omerTT/Calibra
```

---

## Submission Checklist

Before opening any PR, ensure:

- [ ] The repo README shows the CI badge is green
- [ ] `pip install calibra-robotics` works and `calibra --help` prints output
- [ ] The technical report PDF is either committed or linked from the README
- [ ] The experiment notebook `experiments/cross_dataset_analysis.ipynb` is committed

---

## Timing

Awesome List PRs are more likely to be accepted when:
1. The arXiv preprint is live (adds legitimacy)
2. The repo has ≥ 50 stars (social proof)
3. The README includes a quick-start that works in 3 commands

Suggested order:
1. Commit this report + experiment notebook → push to GitHub
2. Post the HuggingFace blog post → share on Twitter/X and robotics Discord servers
3. Submit arXiv preprint
4. Open Awesome List PRs (link the arXiv paper in the PR body)
