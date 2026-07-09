"""
Evaluate a trained NN pricer: regression metrics + financial sanity checks.

This is deliberately separate from train_nn.py: training produces a
checkpoint, evaluation loads it and can be re-run on its own without
retraining (Deliverable: "NN should be compared to binomial prices, not
just to its training loss.").
"""
from __future__ import annotations

import pickle
from typing import Callable

import numpy as np

from src.evaluation.metrics import pricing_metrics


def load_nn_pricer(path: str):
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    from src.ml.models import NumPyAdamMLP
    mlp = NumPyAdamMLP(input_dim=5, hidden_dim=artifact["hidden_dim"], output_dim=1)
    mlp.set_state(artifact["model_state"])

    x_mean, x_std = artifact["x_mean"], artifact["x_std"]
    ranges = artifact["ranges"]

    def predict(S0, K, T, r, sigma) -> float:
        X = np.array([[S0, K, T, r, sigma]], dtype=float)
        X_s = (X - x_mean) / x_std
        return float(mlp.forward(X_s).reshape(-1)[0])

    def predict_batch(X_raw: np.ndarray) -> np.ndarray:
        X_s = (X_raw - x_mean) / x_std
        return mlp.forward(X_s).reshape(-1)

    def is_extrapolation(S0, K, T, r, sigma) -> bool:
        vals = {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma}
        return any(vals[k] < lo or vals[k] > hi for k, (lo, hi) in ranges.items())

    return predict, predict_batch, is_extrapolation, artifact


def intrinsic_put_value(S: float, K: float) -> float:
    return max(K - S, 0.0)


def count_intrinsic_violations(rows) -> list:
    violations = []
    for row in rows:
        intrinsic = intrinsic_put_value(row["S0"], row["K"])
        if row["nn_price"] + 1e-8 < intrinsic:
            violations.append(row)
    return violations


def put_spot_monotonicity_check(predict_fn: Callable, K=100, T=1.0, r=0.05, sigma=0.25):
    """Put price should not rise as spot rises. Returns list of violating
    (S_lo, S_hi, price_lo, price_hi) tuples."""
    spots = np.linspace(60, 140, 41)
    prices = [predict_fn(S0=S, K=K, T=T, r=r, sigma=sigma) for S in spots]
    increases = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1] + 1e-6:
            increases.append((spots[i - 1], spots[i], prices[i - 1], prices[i]))
    return increases


def evaluate(checkpoint_path: str, test_npz_path: str) -> dict:
    predict, predict_batch, is_extrapolation, artifact = load_nn_pricer(checkpoint_path)

    data = np.load(test_npz_path)
    X_test, y_test = data["X"], data["y"].reshape(-1)
    pred_test = predict_batch(X_test)

    overall = pricing_metrics(y_test, pred_test)

    moneyness = X_test[:, 0] / X_test[:, 1]
    buckets = {
        "deep_itm_put": moneyness < 0.85,
        "near_atm": (moneyness >= 0.85) & (moneyness <= 1.15),
        "deep_otm_put": moneyness > 1.15,
    }
    by_bucket = {name: pricing_metrics(y_test[mask], pred_test[mask])
                 for name, mask in buckets.items() if mask.any()}

    rows = [{"S0": X_test[i, 0], "K": X_test[i, 1], "nn_price": pred_test[i]} for i in range(len(X_test))]
    violations = count_intrinsic_violations(rows)
    mono_violations = put_spot_monotonicity_check(lambda **kw: predict(**kw))

    return {
        "overall": overall,
        "by_bucket": by_bucket,
        "n_intrinsic_violations": len(violations),
        "n_monotonicity_violations": len(mono_violations),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="reports/checkpoints/nn_pricer.pkl")
    parser.add_argument("--test-data", type=str, default="reports/checkpoints/nn_pricer_test.npz")
    args = parser.parse_args()

    results = evaluate(args.checkpoint, args.test_data)
    print(json.dumps(results, indent=2))
