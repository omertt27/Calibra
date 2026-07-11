---
name: claim-honesty-pass
description: The 2026-07-11 correction of public overclaims in README/RESULTS and the positioning + next-benchmark decision it locked in
metadata:
  type: project
---

On 2026-07-11 we did a claim-honesty pass on the public docs after an audit found the detailed results were honest but the summaries overstated them. Fixes applied to `README.md` and `experiments/RESULTS.md`:

1. **Demoted the single-seed "4× / 8% vs 2%" PushT headline** to an explicitly labelled *exploratory single-seed case study*. It's within seed noise — the 10-seed sweep shows the clean coreset is 14.8% ± 6.3%. The seeded 5-seed / 3-dataset paired-t ablation is the headline result. See [[coreset-seeded-findings]].
2. **"16 datasets" → "7 verified datasets"** for the ρ = 0.5971 cross-dataset correlation. 16 was the count of *reference profiles*, not correlation points; DROID/BridgeData are excluded from the correlation (control-mode mismatch).
3. **Clarified 95.8% compute figure** as *optimization-step* reduction under an episode-scaled step schedule, NOT a universal training-cost saving from keeping 30% of episodes.
4. **Separated the two Spearman values**: ρ = 0.6749 is within-PushT corruption-severity; ρ = 0.5971 is cross-dataset ranking. Never merge as "0.60–0.67".
5. **Fixed a data error**: RESULTS.md said root-cause 100% (9/9); `results_l4l6.json` says 0.889 (8/9). README was already correct.

**Positioning locked in:** "dataset observability framework whose diversity-based selection is *competitive with* established coresets (not a claimed improvement), with *moderate* pre-training predictive signal." Do NOT pitch as "new quality-aware coreset that beats existing methods" — the seeded paired-t tests show quality-filter loses to its own diversity stage on 2/3 datasets.

**Keep out of the main pitch** (label experimental/hypothesis only): validated sim2real prediction, cross-embodiment transfer, universal contact-aware regime detection, a generally-valid 0–100 score, "millions saved", universal success prediction, world-model superiority.

**Gating next benchmark before outreach/arXiv: run ACT on the same 3 datasets** — tests whether findings survive outside BC-MLP. Then a fair compute benchmark (equal steps/epochs/wall-clock, episode-scaled budget reported separately). Still lower-confidence in the claim registry: PAI-001 (sim2real, only prose evidence), ENT-001 (OOD link unmeasured), JS-002/VD-002 (falsified), VD-001 (5% threshold too tight).