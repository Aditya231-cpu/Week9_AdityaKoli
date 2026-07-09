"""
Train the neural-network pricer on synthetic binomial-labeled contracts.

Usage:
    python -m src.ml.train_nn --out reports/checkpoints/nn_pricer.pkl
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from src.data.synthetic_contracts import make_training_dataset, TRAINING_RANGES
from src.ml.models import NumPyAdamMLP


def train(n_samples: int = 10_000, epochs: int = 200, batch_size: int = 256,
          hidden_dim: int = 256, lr: float = 1e-3, seed: int = 42,
          binomial_steps: int = 500):
    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} synthetic contracts and CRR labels (steps={binomial_steps})...")
    X, y = make_training_dataset(n_samples=n_samples, seed=seed, steps=binomial_steps)
    y = y.reshape(-1, 1)

    # 80/10/10 split
    idx = rng.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)

    X_train_s = (X_train - x_mean) / x_std
    X_val_s = (X_val - x_mean) / x_std

    mlp = NumPyAdamMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=1, seed=seed)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None

    print("Training neural-network pricer...")
    for epoch in range(epochs):
        order = rng.permutation(len(X_train_s))
        batch_losses = []
        for start in range(0, len(X_train_s), batch_size):
            b = order[start:start + batch_size]
            pred = mlp.forward(X_train_s[b])
            mlp.backward(pred, y_train[b])
            mlp.update(lr=lr)
            batch_losses.append(np.mean((pred - y_train[b]) ** 2))

        train_loss = float(np.mean(batch_losses))
        val_pred = mlp.forward(X_val_s)
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val, best_state = val_loss, mlp.get_state()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    mlp.set_state(best_state)
    print(f"Restored best checkpoint. Best Validation MSE: {best_val:.6f}")

    artifact = {
        "model_state": best_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_order": ["S0", "K", "T", "r", "sigma"],
        "label_steps": binomial_steps,
        "ranges": TRAINING_RANGES,
        "hidden_dim": hidden_dim,
        "history": history,
    }
    return artifact, (X_test, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="reports/checkpoints/nn_pricer.pkl")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifact, (X_test, y_test) = train(n_samples=args.n_samples, epochs=args.epochs, seed=args.seed)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved trained NN pricer to {args.out}")

    # also stash the held-out test split next to the checkpoint for evaluate_nn.py
    test_path = os.path.splitext(args.out)[0] + "_test.npz"
    np.savez_compressed(test_path, X=X_test, y=y_test)
    print(f"Saved held-out test split to {test_path}")


if __name__ == "__main__":
    main()
