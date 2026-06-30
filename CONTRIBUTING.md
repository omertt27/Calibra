# Contributing to Calibra

Thank you for your interest in contributing to Calibra! This project aims to bring dataset observability and coreset selection to imitation learning in robotics.

Following these guidelines ensures a smooth contribution process for everyone.

---

## Contributor License Agreement (CLA)

Calibra is licensed under the [Business Source License 1.1](LICENSE). By submitting a pull request, you agree to the following:

1. **You grant the Licensor (omerTT) a perpetual, worldwide, non-exclusive, royalty-free license** to use, reproduce, modify, sublicense, and distribute your contribution as part of Calibra under any license terms the Licensor chooses, including future commercial or open-source licenses.
2. **You confirm that your contribution is your original work** and that you have the right to grant the above license (i.e., it does not include code owned by your employer or a third party without their permission).
3. **You understand that Calibra is not MIT-licensed.** Using or distributing Calibra in violation of the BSL 1.1 terms requires a separate commercial license — contact omertahtoko@gmail.com.

You do not need to sign anything separately. Checking the CLA box in the pull request template is your agreement on record.

For questions about licensing or commercial use, open an issue or email omertahtoko@gmail.com.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

To set up a local development environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/omerTT/Calibra.git
   cd Calibra
   ```

2. **Install in Editable Mode with Dev/Extra dependencies:**
   It is recommended to run in a virtual environment (e.g., `venv` or `conda`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e '.[all,dev]'
   ```

---

## Code Style & Formatting

We use **Ruff** for style checking and formatting. Ensure your code satisfies the rules before submitting a PR.

* **Check for Lint Issues:**
  ```bash
  ruff check .
  ```
* **Auto-format Code:**
  ```bash
  ruff format .
  ```

---

## Running Tests

We use **pytest** to run unit tests. Make sure all tests pass:

```bash
pytest
```

---

## Contribution Workflows

### 1. Profiling a New Dataset (Highly Valued!)
The evidence base for dataset diagnostics grows with every new reference profile. To profile a new dataset (e.g., a LeRobot or HDF5 robot dataset):

1. **Run the Profile Script:**
   ```bash
   python scripts/profile_dataset.py lerobot/my_new_dataset \
     --control-mode position \
     --out calibra/references/my_new_dataset.json \
     --note "Brief description of the arm, real/sim, frequency, etc."
   ```

2. **Register Evidence:**
   If the dataset satisfies or falsifies any active assertions in `calibra/claims/*.json`, open the relevant JSON file and add your dataset to the `evidence` array.

3. **Regenerate Claims Documentation:**
   ```bash
   python scripts/generate_claims_doc.py
   ```

### 2. Updating Claims and Assertions
All dataset diagnostic assertions are stored in `calibra/claims/*.json`.

- When adding a new claim, follow the structure specified in `calibra/claims/SPEC.md`.
- **Claim-to-Profile Ratio Rule:** To prevent ungrounded theories, the number of reference profiles in the project must always be greater than or equal to the number of active claims.
- Verify your changes comply with the ratio rule before committing:
  ```bash
  python scripts/generate_claims_doc.py --check
  ```

---

## Creating a Pull Request

1. **Branch Naming:** Use descriptive branch names like `feature/add-profile-so100` or `bugfix/fix-ldlj-nan`.
2. **Commit Messages:** Keep messages concise and clear (e.g., `feat: profile Bridgedata v3 at 5Hz`).
3. **Verify:** Ensure `pytest`, `ruff check .`, and `python scripts/generate_claims_doc.py --check` all pass.
