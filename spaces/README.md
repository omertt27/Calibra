---
title: Robot Dataset Health Check
emoji: 🤖
colorFrom: violet
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: bsl-1.1
short_description: Audit any LeRobot dataset for quality issues in 30 seconds
tags:
  - robotics
  - dataset-quality
  - lerobot
  - imitation-learning
  - data-curation
---

# Robot Dataset Health Check

Audit any [LeRobot](https://github.com/huggingface/lerobot) dataset in ~30 seconds.

Enter a dataset ID (e.g. `lerobot/pusht`) and get:

- **0–100 health score** with grade (A–F) and certification status
- **Concrete findings** — frame dropout count, jerk trajectory count, redundant episodes
- **Keep-fraction recommendation** — how much of the dataset to train on and which strategy to use
- **Dimension breakdown** — temporal, smoothness, coverage, task structure, dynamics
- **Downloadable CalibraReport JSON** for CI pipelines and reproducibility

## Run locally

```bash
pip install 'calibra-robotics[lerobot]'
calibra audit lerobot/pusht
```

## Community benchmark

See [calibra-robot-dataset-quality-benchmark](https://huggingface.co/datasets/omert27/calibra-robot-dataset-quality-benchmark)
for audits of 30+ public LeRobot datasets with a sortable leaderboard.

## About

Powered by [Calibra](https://github.com/omertt27/Calibra) — open-source dataset
quality tooling for robotics imitation learning.
