#!/bin/bash
# reproduce_results.sh
# One script to regenerate all benchmark figures, correlation plots, and paper figures.
# Reproduces all quantitative results in the Calibra paper.
#
# Usage:
#   ./experiments/reproduce_results.sh            # full replication
#   ./experiments/reproduce_results.sh --fast     # skip scale benchmark (slow)

set -e

FAST=0
for arg in "$@"; do
  if [ "$arg" = "--fast" ]; then FAST=1; fi
done

echo "=== Calibra v0.6.0 Results Replication Package ==="
echo ""

# ── Environment check ─────────────────────────────────────────────────────────
echo "Python environment check:"
python -c "import numpy, pandas, scipy, matplotlib; print('  ✓ Core dependencies present')"
python -c "import torch; print('  ✓ PyTorch present')" 2>/dev/null || echo "  ⚠ PyTorch not found (needed for real-data benchmark)"
python -c "import gym_pusht, gymnasium; print('  ✓ gym-pusht present')" 2>/dev/null || echo "  ⚠ gym-pusht not found (needed for real-data benchmark — pip install gym-pusht gymnasium)"

# ── Phase 1: Real-data coreset benchmark ──────────────────────────────────────
echo ""
echo "=== Phase 1: Real-Data Coreset Benchmark (gym_pusht/PushT-v0) ==="
echo "  Collects 500 scripted-expert episodes, curates with Calibra, trains BC."
echo "  Requires: pip install gym-pusht gymnasium 'pymunk==6.9.0'"
echo "  Expected: Calibra coreset SR>=50% > random baseline SR>=50%"
echo "  Expected: Calibra coreset compute savings > 90%"
PYTHONPATH=. python experiments/pusht_real_benchmark.py

# ── Phase 2: Predictor correlation study ─────────────────────────────────────
echo ""
echo "=== Phase 2: Predictor Correlation Study (16 datasets) ==="
echo "  Generates: experiments/figures/fig6_predict_correlation.pdf"
echo "  Expected:  Spearman rho ≈ 0.597, p < 0.05"
PYTHONPATH=. python experiments/predict_correlation_study.py

# ── Phase 3: Scale benchmark ──────────────────────────────────────────────────
if [ "$FAST" -eq 0 ]; then
  echo ""
  echo "=== Phase 3: Approximate Coreset Scale Benchmark ==="
  echo "  Generates: experiments/figures/fig_scale_benchmark.pdf"
  echo "  Tests N = 1k, 5k, 10k, 50k, 100k, 500k episodes"
  echo "  (Takes ~5 min on a single CPU core — use --fast to skip)"
  PYTHONPATH=. python experiments/scale_benchmark.py --max-n 500000 --save-fig
else
  echo ""
  echo "=== Phase 3: Scale Benchmark (SKIPPED — --fast mode) ==="
fi

# ── Phase 4: Paper figures ────────────────────────────────────────────────────
echo ""
echo "=== Phase 4: Paper Figures ==="
echo "  Generates: experiments/figures/fig_system_overview.pdf"
PYTHONPATH=. python scripts/generate_paper_figures.py

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Results regenerated successfully ==="
echo ""
echo "Output files:"
ls -1 experiments/figures/*.pdf 2>/dev/null | sed 's/^/  /'
echo ""
echo "To compile the paper:"
echo "  cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
echo ""
echo "Key numbers to verify:"
echo "  Spearman rho >= 0.59          (predictor correlation)"
echo "  Calibra SR>=50%  >= 6%        (vs. 2% full-dataset BC)"
echo "  Calibra compute savings >= 90% (vs. 66% random-30% baseline)"
echo "  500k episodes in < 120s       (approximate selector)"
