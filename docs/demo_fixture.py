"""
Generates a small synthetic HDF5 dataset with an injected short episode, a
frozen camera episode, and a blurry episode, so `calibra integrity` has
something real to flag on all three image checks plus the timestamp checks.
Used by docs/demo.tape to produce a reproducible demo recording without
depending on a real dataset being available.

Actions are a smooth low-frequency sinusoid rather than i.i.d. noise: raw
per-step random values are maximally jittery by construction (velocity
reverses every step), which trips the ControlSmoothnessAnalyzer jerk/
velocity-discontinuity checks that are also part of `calibra integrity` —
that noise would swamp the specific defects this fixture is meant to
demonstrate.

Usage:
    python docs/demo_fixture.py /tmp/calibra_demo.h5
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


def _smooth_actions(rng: np.random.Generator, steps: int, dt: float, dims: int = 6) -> np.ndarray:
    """Low-frequency sinusoid trajectory — smooth by construction (bounded
    jerk), unlike i.i.d. per-step noise."""
    t = np.arange(steps) * dt
    actions = np.zeros((steps, dims), dtype=np.float32)
    for d in range(dims):
        for h in range(1, 3):
            freq = h * rng.uniform(0.05, 0.12)
            amp = rng.uniform(0.05, 0.15) / h
            phase = rng.uniform(0, 2 * np.pi)
            actions[:, d] += amp * np.sin(2 * np.pi * freq * t + phase)
        actions[:, d] += rng.uniform(-0.3, 0.3)
    return actions.astype(np.float32)


def main(path: str) -> None:
    rng = np.random.default_rng(0)
    n_steps = 40
    with h5py.File(path, "w") as f:
        for i in range(8):
            g = f.create_group(f"episode_{i}")
            obs = g.create_group("observations")
            steps = 5 if i == 7 else n_steps  # one short episode

            if i == 4:
                # Blurry episode: smooth gradient + tiny noise so frames still
                # differ from each other (avoids also tripping duplicate/freeze).
                x = np.linspace(0, 1, 16)
                gradient = (np.outer(x, x) * 255).astype(np.uint8)
                frame = np.stack([gradient] * 3, axis=-1)
                frames = np.tile(frame, (steps, 1, 1, 1)).astype(np.uint8)
                frames = frames + rng.integers(0, 3, frames.shape, dtype=np.uint8)
            else:
                frames = rng.integers(0, 255, (steps, 16, 16, 3), dtype=np.uint8)
            if i == 0:
                frames[:12] = frames[0]  # one frozen-camera episode

            obs.create_dataset("camera_rgb", data=frames)
            obs.create_dataset("proprio", data=rng.random((steps, 8)).astype(np.float32))
            g.create_dataset("actions", data=_smooth_actions(rng, steps, dt=0.05))
            g.create_dataset("timestamps", data=np.arange(steps) * 0.05)
    print(f"Wrote demo dataset to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/calibra_demo.h5")
