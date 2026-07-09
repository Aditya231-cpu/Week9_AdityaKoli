"""
Generate the required final figures from the project spec:
  1. Payoff diagram
  2. Binomial exercise boundary plot
  3. NN predicted vs binomial scatter
  4. NN error slice/heatmap by moneyness and maturity
  5. RL exercise-region plot over time and S/K
  6. (Comparison table is produced by comparison.py, not here)

Usage:
    python -m src.evaluation.plots --out reports/figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pricing.binomial import OptionContract, crr_american_put_with_boundary
from src.pricing.payoffs import put_payoff, call_payoff


def plot_payoff_diagram(out_dir: str, K: float = 100.0):
    S = np.linspace(0, 2 * K, 200)
    put = [put_payoff(s, K) for s in S]
    call = [call_payoff(s, K) for s in S]

    plt.figure(figsize=(6, 4.5))
    plt.plot(S, put, label="Put payoff: max(K-S, 0)", color="crimson")
    plt.plot(S, call, label="Call payoff: max(S-K, 0)", color="royalblue")
    plt.axvline(K, color="gray", linestyle=":", label=f"Strike K={K:.0f}")
    plt.xlabel("Spot price S")
    plt.ylabel("Payoff")
    plt.title("Option payoff at expiry")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_payoff_diagram.png"), dpi=160)
    plt.close()


def plot_exercise_boundary_binomial(out_dir: str, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=200):
    contract = OptionContract(S0=100.0, K=K, T=T, r=r, sigma=sigma, steps=steps, option_type="put")
    _, boundary = crr_american_put_with_boundary(contract)

    steps_sorted = sorted(boundary.keys())
    times = [contract.T * s / steps for s in steps_sorted]
    spots = [boundary[s] for s in steps_sorted]

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(times, spots, color="darkorange", linewidth=2)
    plt.fill_between(times, 0, spots, color="darkorange", alpha=0.15, label="Exercise region")
    plt.xlabel("Time (years)")
    plt.ylabel("Critical spot price")
    plt.title(f"CRR American put exercise boundary\n(K={K}, T={T}, r={r}, sigma={sigma})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_binomial_exercise_boundary.png"), dpi=160)
    plt.close()


def plot_nn_vs_binomial_scatter(df: pd.DataFrame, out_dir: str):
    x, y = df["binomial_price"], df["nn_price"]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.6, s=20)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="Parity")
    plt.xlabel("Binomial benchmark price")
    plt.ylabel("NN predicted price")
    plt.title("NN price vs binomial benchmark")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_nn_vs_binomial_scatter.png"), dpi=160)
    plt.close()


def plot_nn_error_heatmap(df: pd.DataFrame, out_dir: str):
    pivot = df.pivot_table(index="sigma", columns="T", values="nn_error", aggfunc="mean")
    plt.figure(figsize=(6.5, 5))
    im = plt.imshow(pivot.values, cmap="coolwarm", aspect="auto",
                     vmin=-abs(pivot.values).max(), vmax=abs(pivot.values).max())
    plt.xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{i:.2f}" for i in pivot.index])
    plt.xlabel("Maturity T")
    plt.ylabel("Volatility sigma")
    plt.title("Mean NN error (NN - binomial) by T x sigma")
    plt.colorbar(im, label="Mean error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_nn_error_heatmap.png"), dpi=160)
    plt.close()


def plot_rl_exercise_region(policy_fn, out_dir: str, steps=50, money_min=0.7, money_max=1.2, n_points=100):
    time_steps = np.linspace(0, steps, n_points)
    moneyness_values = np.linspace(money_min, money_max, n_points)
    grid = np.zeros((n_points, n_points))

    for t_idx, t in enumerate(time_steps):
        for m_idx, m in enumerate(moneyness_values):
            state = np.array([t / steps, 1.0 - t / steps, m], dtype=np.float32)
            grid[m_idx, t_idx] = policy_fn(state)

    plt.figure(figsize=(8, 5.5))
    plt.imshow(grid, origin="lower", aspect="auto",
               extent=[0, 1, money_min, money_max],
               cmap=matplotlib.colors.ListedColormap(["#1f77b4", "#ff7f0e"]))
    plt.xlabel("Time fraction (t/T)")
    plt.ylabel("Moneyness (S/K)")
    plt.title("RL (DQN) learned exercise region")
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor="#1f77b4", label="Hold"),
        Patch(facecolor="#ff7f0e", label="Exercise"),
    ], loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_rl_exercise_region.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="reports/figures")
    parser.add_argument("--comparison-csv", type=str, default="reports/comparison.csv")
    parser.add_argument("--dqn-checkpoint", type=str, default="reports/checkpoints/dqn.pth")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    plot_payoff_diagram(args.out)
    plot_exercise_boundary_binomial(args.out)

    if os.path.exists(args.comparison_csv):
        df = pd.read_csv(args.comparison_csv)
        plot_nn_vs_binomial_scatter(df, args.out)
        plot_nn_error_heatmap(df, args.out)
    else:
        print(f"Skipping NN plots: {args.comparison_csv} not found. Run comparison.py first.")

    if os.path.exists(args.dqn_checkpoint):
        import torch
        from src.rl.train_dqn import QNetwork, make_dqn_policy
        online = QNetwork()
        online.load_state_dict(torch.load(args.dqn_checkpoint, map_location="cpu"))
        online.eval()
        plot_rl_exercise_region(make_dqn_policy(online), args.out)
    else:
        print(f"Skipping RL plot: {args.dqn_checkpoint} not found. Run train_dqn.py first.")

    print(f"Figures written to {args.out}")
