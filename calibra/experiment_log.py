"""
calibra.experiment_log — Local record of design-partner training experiments.

Implements the bookkeeping side of the design-partner protocol: every pilot
runs the same three conditions (full dataset, random subset, Calibra coreset)
at multiple retention levels, and the *comparison itself* only means something
if the numbers behind it are recorded consistently.

Unlike calibra.outcome_db, this store never syncs to any network endpoint.
Design-partner datasets and training results are the customer's most sensitive
asset — the whole pitch is that Calibra runs inside their infrastructure and
their data never has to leave it. Keep that true here too.

Storage: JSON Lines, one record per (experiment_id, condition, retention_pct)
observation, at ~/.calibra/experiments.jsonl by default.

Usage
-----
    from calibra.experiment_log import ExperimentLog

    log = ExperimentLog()
    log.record(
        experiment_id="partner-a-pusht",
        dataset_name="partner-a/pusht_v3",
        condition="calibra",
        retention_pct=25.0,
        n_episodes=300,
        policy_family="act",
        eval_success_rate=0.84,
        gpu_hours=19.8,
        wall_clock_seconds=71280.0,
        seed=0,
    )

    table = log.retention_table("partner-a-pusht")
    print(log.report("partner-a-pusht"))
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

CONDITIONS = ("full", "random", "calibra")

# The design-partner protocol's minimum retention sweep (see project memory:
# project_calibra_design_partner_protocol). A single impressive ratio isn't
# enough evidence — this is the curve a skeptical reviewer expects to see.
PROTOCOL_RETENTION_LEVELS = (10.0, 25.0, 50.0, 75.0, 100.0)


class ExperimentRecord:
    __slots__ = (
        "record_id",
        "timestamp",
        "experiment_id",
        "partner",
        "dataset_name",
        "embodiment",
        "task",
        "policy_family",
        "model_size",
        "condition",
        "retention_pct",
        "n_episodes",
        "gpu_hours",
        "wall_clock_seconds",
        "energy_kwh",
        "training_loss",
        "eval_success_rate",
        "seed",
        "notes",
    )

    def __init__(
        self,
        record_id: str,
        timestamp: float,
        experiment_id: str,
        condition: str,
        retention_pct: float,
        dataset_name: str = "unknown",
        partner: str = "",
        embodiment: str = "",
        task: str = "",
        policy_family: str = "generic",
        model_size: str = "",
        n_episodes: int = 0,
        gpu_hours: Optional[float] = None,
        wall_clock_seconds: Optional[float] = None,
        energy_kwh: Optional[float] = None,
        training_loss: Optional[float] = None,
        eval_success_rate: Optional[float] = None,
        seed: Optional[int] = None,
        notes: str = "",
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        if not 0.0 <= retention_pct <= 100.0:
            raise ValueError(f"retention_pct must be in [0, 100], got {retention_pct!r}")
        if eval_success_rate is not None and not 0.0 <= eval_success_rate <= 1.0:
            raise ValueError(
                f"eval_success_rate must be in [0, 1], got {eval_success_rate!r}"
            )

        self.record_id = record_id
        self.timestamp = timestamp
        self.experiment_id = experiment_id
        self.partner = partner
        self.dataset_name = dataset_name
        self.embodiment = embodiment
        self.task = task
        self.policy_family = policy_family
        self.model_size = model_size
        self.condition = condition
        self.retention_pct = retention_pct
        self.n_episodes = n_episodes
        self.gpu_hours = gpu_hours
        self.wall_clock_seconds = wall_clock_seconds
        self.energy_kwh = energy_kwh
        self.training_loss = training_loss
        self.eval_success_rate = eval_success_rate
        self.seed = seed
        self.notes = notes

    def to_dict(self) -> dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        return cls(
            record_id=d["record_id"],
            timestamp=d["timestamp"],
            experiment_id=d["experiment_id"],
            condition=d["condition"],
            retention_pct=d["retention_pct"],
            dataset_name=d.get("dataset_name", "unknown"),
            partner=d.get("partner", ""),
            embodiment=d.get("embodiment", ""),
            task=d.get("task", ""),
            policy_family=d.get("policy_family", "generic"),
            model_size=d.get("model_size", ""),
            n_episodes=d.get("n_episodes", 0),
            gpu_hours=d.get("gpu_hours"),
            wall_clock_seconds=d.get("wall_clock_seconds"),
            energy_kwh=d.get("energy_kwh"),
            training_loss=d.get("training_loss"),
            eval_success_rate=d.get("eval_success_rate"),
            seed=d.get("seed"),
            notes=d.get("notes", ""),
        )


_DEFAULT_DB_PATH = Path.home() / ".calibra" / "experiments.jsonl"


class ExperimentLog:
    """
    Append-only local store of design-partner experiment observations.

    No network calls anywhere in this class — that's deliberate, see module
    docstring. If cloud sync of experiment results is ever wanted, it must be
    a separate, explicit, opt-in action, never a side effect of recording.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _DEFAULT_DB_PATH
        self._records: list[ExperimentRecord] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(ExperimentRecord.from_dict(json.loads(line)))
                except Exception:
                    pass

    def _append(self, rec: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    # ── public API ────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        experiment_id: str,
        condition: str,
        retention_pct: float,
        dataset_name: str = "unknown",
        partner: str = "",
        embodiment: str = "",
        task: str = "",
        policy_family: str = "generic",
        model_size: str = "",
        n_episodes: int = 0,
        gpu_hours: Optional[float] = None,
        wall_clock_seconds: Optional[float] = None,
        energy_kwh: Optional[float] = None,
        training_loss: Optional[float] = None,
        eval_success_rate: Optional[float] = None,
        seed: Optional[int] = None,
        notes: str = "",
    ) -> ExperimentRecord:
        rec = ExperimentRecord(
            record_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            experiment_id=experiment_id,
            condition=condition,
            retention_pct=retention_pct,
            dataset_name=dataset_name,
            partner=partner,
            embodiment=embodiment,
            task=task,
            policy_family=policy_family,
            model_size=model_size,
            n_episodes=n_episodes,
            gpu_hours=gpu_hours,
            wall_clock_seconds=wall_clock_seconds,
            energy_kwh=energy_kwh,
            training_loss=training_loss,
            eval_success_rate=eval_success_rate,
            seed=seed,
            notes=notes,
        )
        self._records.append(rec)
        self._append(rec)
        return rec

    def experiments(self) -> list[str]:
        """Return the distinct experiment_ids present, in first-seen order."""
        seen: list[str] = []
        for rec in self._records:
            if rec.experiment_id not in seen:
                seen.append(rec.experiment_id)
        return seen

    def list_records(self, experiment_id: Optional[str] = None) -> list[ExperimentRecord]:
        if experiment_id is None:
            return list(self._records)
        return [r for r in self._records if r.experiment_id == experiment_id]

    def retention_table(self, experiment_id: str) -> dict[float, dict[str, ExperimentRecord]]:
        """
        Return {retention_pct: {condition: record}} for one experiment.

        If multiple records share the same (retention_pct, condition), the
        most recently recorded one wins — re-running a condition is assumed
        to supersede the prior attempt, not average with it.
        """
        table: dict[float, dict[str, ExperimentRecord]] = {}
        for rec in self.list_records(experiment_id):
            table.setdefault(rec.retention_pct, {})[rec.condition] = rec
        return table

    def missing_conditions(self, experiment_id: str) -> list[tuple[float, str]]:
        """
        Return (retention_pct, condition) pairs the protocol expects but that
        haven't been recorded yet, against PROTOCOL_RETENTION_LEVELS.
        """
        table = self.retention_table(experiment_id)
        missing = []
        for level in PROTOCOL_RETENTION_LEVELS:
            have = table.get(level, {})
            for cond in CONDITIONS:
                if cond == "full" and level != 100.0:
                    continue  # "full" only makes sense at 100% retention
                if cond != "full" and level == 100.0:
                    continue  # random/calibra at 100% retention is a no-op
                if cond not in have:
                    missing.append((level, cond))
        return missing

    def calibra_vs_random(self, experiment_id: str) -> dict[float, Optional[float]]:
        """
        Return {retention_pct: eval_success_rate delta (calibra - random)} for
        every retention level where both conditions have a recorded eval
        success rate. None where either side is missing or unmeasured.
        """
        table = self.retention_table(experiment_id)
        deltas: dict[float, Optional[float]] = {}
        for level, conditions in table.items():
            if level == 100.0:
                continue
            calibra_rec = conditions.get("calibra")
            random_rec = conditions.get("random")
            if (
                calibra_rec is None
                or random_rec is None
                or calibra_rec.eval_success_rate is None
                or random_rec.eval_success_rate is None
            ):
                deltas[level] = None
                continue
            deltas[level] = calibra_rec.eval_success_rate - random_rec.eval_success_rate
        return deltas

    def report(self, experiment_id: str) -> str:
        """Render a human-readable retention-curve report for one experiment."""
        table = self.retention_table(experiment_id)
        if not table:
            return f"No records for experiment_id={experiment_id!r}."

        lines = [f"Experiment: {experiment_id}", "─" * 72]
        header = f"{'Retention':>10} {'Cond':<8} {'Eps':>7} {'Success':>9} {'GPU-hrs':>9} {'Wall-clk':>10}"
        lines.append(header)
        for level in sorted(table.keys()):
            for cond in CONDITIONS:
                rec = table[level].get(cond)
                if rec is None:
                    continue
                success = f"{rec.eval_success_rate:.1%}" if rec.eval_success_rate is not None else "n/a"
                gpu = f"{rec.gpu_hours:.1f}" if rec.gpu_hours is not None else "n/a"
                wall = f"{rec.wall_clock_seconds / 3600:.1f}h" if rec.wall_clock_seconds is not None else "n/a"
                lines.append(
                    f"{level:>9.0f}% {cond:<8} {rec.n_episodes:>7} {success:>9} {gpu:>9} {wall:>10}"
                )

        deltas = self.calibra_vs_random(experiment_id)
        measured = {k: v for k, v in deltas.items() if v is not None}
        if measured:
            lines.append("─" * 72)
            for level in sorted(measured):
                sign = "beats" if measured[level] > 0 else "trails" if measured[level] < 0 else "ties"
                lines.append(
                    f"  At {level:.0f}% retention: Calibra {sign} random by "
                    f"{abs(measured[level]):.1%} eval success."
                )

        missing = self.missing_conditions(experiment_id)
        if missing:
            lines.append("─" * 72)
            lines.append(f"  Protocol incomplete — {len(missing)} condition(s) not yet recorded:")
            for level, cond in missing:
                lines.append(f"    {level:.0f}% / {cond}")
        else:
            lines.append("─" * 72)
            lines.append("  Protocol complete: full baseline + random/calibra at all retention levels.")

        return "\n".join(lines)

    def summary(self) -> str:
        n = len(self._records)
        if n == 0:
            return f"Experiment log: 0 records at {self.path}."
        n_exp = len(self.experiments())
        return (
            f"Experiment log: {n} record(s) across {n_exp} experiment(s) at {self.path}\n"
            f"  Experiments: {', '.join(self.experiments())}"
        )
