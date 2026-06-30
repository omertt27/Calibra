## Description

Please include a summary of the changes and the related issue/motivation. If applicable, specify which components (analyzers, core, metrics, ingestion adapters) are affected.

Fixes # (issue number)

## Type of Change

Please delete options that are not relevant:

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] New dataset profile (added JSON reference under `references/` and updated evidence)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update (e.g. updating interpretations, claims, guides)

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce.

- **Unit tests run:** `pytest` output or specific command
- **Dataset verification (if applicable):** did you test against local sample HDF5/Parquet files?
- **Claims doc check:** did `python scripts/generate_claims_doc.py --check` pass?

## Contributor License Agreement

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md) and agree to the Calibra CLA. I confirm this contribution is my original work and I grant the Licensor the rights described there.

## Checklist

- [ ] My code follows the style guidelines of this project (ran `ruff check .` and `ruff format .`)
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules
