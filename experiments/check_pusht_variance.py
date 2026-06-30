import sys
import os
import numpy as np
import torch
import random

sys.path.append(os.getcwd())

from experiments.failure_prevention_benchmark import (
    collect_base_data,
    select_calibra_coreset,
    build_condition_batch,
    train_bc,
    evaluate_bc,
    DEVICE,
    SEED,
    KEEP_FRACTION,
)

def main():
    print("Collecting base data...")
    base_batch = collect_base_data(500, SEED)
    print("Selecting Calibra coreset...")
    calibra_idx = select_calibra_coreset(base_batch, KEEP_FRACTION)

    seeds = list(range(42, 52)) # 10 seeds
    clean_srs = []
    spike_srs = []

    print("\n--- Running 10 seeds ---")
    print(f"{'Seed':<6} | {'Clean SR':<10} | {'Spike 2% SR':<12}")
    print("-" * 35)

    for seed in seeds:
        # Use a fixed rng for the corruption to make the dataset identical across seeds,
        # so the only variance is the training seed.
        rng_corr = np.random.default_rng(42)
        clean_batch = build_condition_batch(base_batch, calibra_idx, "none", 0.0, rng_corr)
        
        rng_corr = np.random.default_rng(42)
        spike_batch = build_condition_batch(base_batch, calibra_idx, "spike", 0.02, rng_corr)

        # Train and eval Clean
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        policy_clean, _ = train_bc(clean_batch)
        sr_clean = evaluate_bc(policy_clean, n_eval=100)

        # Train and eval Spike 2%
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        policy_spike, _ = train_bc(spike_batch)
        sr_spike = evaluate_bc(policy_spike, n_eval=100)

        clean_srs.append(sr_clean)
        spike_srs.append(sr_spike)

        print(f"{seed:<6} | {sr_clean:<10.1%} | {sr_spike:<12.1%}", flush=True)

    print("-" * 35)
    print(f"Clean Mean: {np.mean(clean_srs):.1%} +/- {np.std(clean_srs):.1%}")
    print(f"Spike Mean: {np.mean(spike_srs):.1%} +/- {np.std(spike_srs):.1%}")

if __name__ == "__main__":
    main()
