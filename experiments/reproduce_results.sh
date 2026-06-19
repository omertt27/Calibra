#!/bin/bash
# reproduce_results.sh
# One script to regenerate all benchmark figures and correlation plots.

set -e

echo "=== Calibra Results Replication Package ==="
echo "Python environment check:"
python -c "import torch, numpy, pandas, scipy, matplotlib; print('✓ All dependencies present')"

echo ""
echo "=== Phase 1: Running Curation Performance Benchmark ==="
PYTHONPATH=. python experiments/prune_performance_benchmark.py

echo ""
echo "=== Phase 2: Running Predictor Correlation Study ==="
PYTHONPATH=. python experiments/predict_correlation_study.py

echo ""
echo "=== Results regenerated successfully ==="
echo "Output files created in experiments/figures/:"
ls -la experiments/figures/
