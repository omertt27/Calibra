# docs/report/README.md

## Technical Report

`calibra_report.tex` is the LaTeX source for the Calibra technical report.

### Building the PDF

Requirements: a standard TeX distribution (TeX Live, MiKTeX, or MacTeX).

```bash
cd docs/report

# Build the summary table first
python scripts/summarize_experiments.py --format latex --out experiments/figures/

# Compile (run twice for cross-references)
pdflatex calibra_report.tex
pdflatex calibra_report.tex
```

Output: `calibra_report.pdf`

### Submitting to arXiv

1. Run `python scripts/summarize_experiments.py --format latex --out experiments/figures/`
   to generate the table included by `\input{}`.
2. Zip `calibra_report.tex` + `experiments/figures/table1_cross_dataset.tex` +
   any figures you want to include from `experiments/figures/*.pdf`.
3. Upload to arXiv under **cs.RO** (primary) and **cs.LG** (secondary).
4. Suggested subject line: *"Calibra: Dataset Observability and Coreset Selection for Robot Imitation Learning"*

### Overleaf

Alternatively, upload `calibra_report.tex` and the `table1_cross_dataset.tex`
file to a new Overleaf project.  Set the main document to `calibra_report.tex`.
