"""
Synthetic contract generation.

Two distinct things live here, and it matters that they stay separate:

1. `make_contract_grid()` - the canonical SHARED evaluation grid used to
   compare binomial vs NN vs RL fairly (same contracts, all methods).
2. `make_training_dataset()` - a wider random sample used only to TRAIN the
   neural network. It should not simply be the evaluation grid, otherwise
   the NN's held-out test error is meaningless.
"""
from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np
import pandas as pd

from src.pricing.binomial import OptionContract, crr_price


def make_contract_grid() -> pd.DataFrame:
    """Canonical shared evaluation grid (see project spec).

    Every method (binomial, NN, RL) is evaluated on exactly this grid so
    that any differences in results come from the method, not the inputs.
    """
    spots = [70, 80, 90, 100, 110, 120, 130]
    strikes = [100]
    maturities = [0.25, 0.5, 1.0, 2.0]
    rates = [0.02, 0.05]
    sigmas = [0.15, 0.25, 0.40]

    rows = []
    for S0, K, T, r, sigma in itertools.product(spots, strikes, maturities, rates, sigmas):
        rows.append({
            "S0": S0,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "steps": 100,
        })
    return pd.DataFrame(rows)


def add_moneyness_bucket(df: pd.DataFrame) -> pd.DataFrame:
    ratio = df["S0"] / df["K"]
    df = df.copy()
    df["bucket"] = pd.cut(
        ratio,
        bins=[0.0, 0.9, 1.1, 10.0],
        labels=["ITM put", "ATM", "OTM put"],
    )
    return df


def make_training_dataset(n_samples: int = 10_000, seed: int = 42,
                           steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Wider random sample of contracts, priced with the CRR benchmark,
    used to train the neural-network pricer (Week 6 origin).

    Ranges are intentionally broader than the evaluation grid so a model
    trained here is not simply memorizing the grid, and so the evaluation
    grid can also probe the edge of the training range (Deliverable:
    "Predictions outside the training range should be labeled as
    extrapolation.").
    """
    rng = np.random.default_rng(seed)
    S0 = rng.uniform(60.0, 140.0, size=n_samples)
    K = rng.uniform(80.0, 120.0, size=n_samples)
    T = rng.uniform(0.05, 2.0, size=n_samples)
    r = rng.uniform(0.00, 0.10, size=n_samples)
    sigma = rng.uniform(0.10, 0.50, size=n_samples)
    X = np.column_stack([S0, K, T, r, sigma])

    y = np.empty(n_samples)
    for i in range(n_samples):
        contract = OptionContract(S0=X[i, 0], K=X[i, 1], T=X[i, 2], r=X[i, 3],
                                   sigma=X[i, 4], steps=steps, option_type="put")
        y[i] = crr_price(contract, american=True)

    # Sanity checks (Week 6 label hygiene, kept here so bad data never
    # silently reaches training).
    intrinsic = np.maximum(X[:, 1] - X[:, 0], 0.0)
    assert np.isfinite(y).all(), "Non-finite label found in synthetic dataset."
    assert (y >= -1e-10).all(), "Negative label found in synthetic dataset."
    assert (y + 1e-8 >= intrinsic).all(), "Label violates American intrinsic-value bound."

    return X, y


TRAINING_RANGES = {
    "S0": [60.0, 140.0], "K": [80.0, 120.0], "T": [0.05, 2.0],
    "r": [0.0, 0.10], "sigma": [0.10, 0.50],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the shared evaluation grid as CSV.")
    parser.add_argument("--out", type=str, default="data/contracts.csv")
    parser.add_argument("--n-training", type=int, default=0,
                         help="If > 0, also generate an NN training dataset (.npz) alongside the grid.")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid = make_contract_grid()
    grid.to_csv(args.out, index=False)
    print(f"Wrote shared evaluation grid ({len(grid)} contracts) to {args.out}")

    if args.n_training > 0:
        X, y = make_training_dataset(n_samples=args.n_training)
        npz_path = os.path.splitext(args.out)[0] + "_training.npz"
        np.savez_compressed(npz_path, X=X, y=y)
        print(f"Wrote {args.n_training} training samples to {npz_path}")
