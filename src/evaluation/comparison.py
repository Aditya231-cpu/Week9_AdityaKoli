"""
Build the single unified comparison table: binomial vs NN vs RL, on the
shared evaluation grid. This is the "one script that regenerates the main
comparison table" required by the project layout.

Usage:
    python -m src.evaluation.comparison --out reports/comparison.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.binomial import OptionContract, BinomialAmericanPutPricer
from src.data.synthetic_contracts import make_contract_grid, add_moneyness_bucket


def compare_one_contract(contract: OptionContract, binomial_pricer, nn_predict_fn, rl_eval_fn) -> dict:
    binomial_result = binomial_pricer.price(contract)
    nn_price = nn_predict_fn(contract)
    rl_result = rl_eval_fn(contract)

    return {
        "S0": contract.S0, "K": contract.K, "T": contract.T,
        "r": contract.r, "sigma": contract.sigma,
        "binomial_price": binomial_result.price,
        "nn_price": nn_price,
        "nn_error": nn_price - binomial_result.price,
        "rl_value": rl_result["value"],
        "rl_std_error": rl_result["std_error"],
        "rl_exercise_rate": rl_result["exercise_rate"],
    }


def summarize_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("bucket", observed=True)
    return grouped.agg(
        nn_mae=("nn_error", lambda x: float(np.mean(np.abs(x)))),
        nn_bias=("nn_error", "mean"),
        rl_value_mean=("rl_value", "mean"),
        binomial_mean=("binomial_price", "mean"),
        count=("bucket", "size"),
    ).reset_index()


def run_comparison(nn_predict_fn, rl_eval_fn, out_path: str = "reports/comparison.csv") -> pd.DataFrame:
    """nn_predict_fn(contract) -> float
    rl_eval_fn(contract) -> dict with keys value/std_error/exercise_rate

    Note: the RL policy is trained on fixed (S0=100, K=100, sigma=0.20)
    dynamics (see src/rl/env.py), so rl_eval_fn should re-parameterize the
    environment per contract, or clearly document that RL values are only
    comparable at matching contract settings. See reports/ for discussion.
    """
    grid = make_contract_grid()
    binomial_pricer = BinomialAmericanPutPricer()

    rows = []
    for _, row in grid.iterrows():
        # NOTE: use row["T"], not row.T -- pandas Series.T is the (no-op)
        # transpose attribute and silently shadows a column literally named "T".
        contract = OptionContract(S0=row["S0"], K=row["K"], T=row["T"], r=row["r"],
                                   sigma=row["sigma"], steps=int(row["steps"]), option_type="put")
        rows.append(compare_one_contract(contract, binomial_pricer, nn_predict_fn, rl_eval_fn))

    df = pd.DataFrame(rows)
    df = add_moneyness_bucket(df)

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote unified comparison table ({len(df)} rows) to {out_path}")

    summary = summarize_by_bucket(df)
    summary_path = os.path.splitext(out_path)[0] + "_by_bucket.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote bucketed summary to {summary_path}")

    return df


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="reports/comparison.csv")
    parser.add_argument("--nn-checkpoint", type=str, default="reports/checkpoints/nn_pricer.pkl")
    parser.add_argument("--dqn-checkpoint", type=str, default="reports/checkpoints/dqn.pth")
    parser.add_argument("--rl-episodes", type=int, default=2000)
    args = parser.parse_args()

    from src.ml.evaluate_nn import load_nn_pricer
    predict, predict_batch, is_extrapolation, nn_artifact = load_nn_pricer(args.nn_checkpoint)

    def nn_predict_fn(contract: OptionContract) -> float:
        return predict(contract.S0, contract.K, contract.T, contract.r, contract.sigma)

    # RL: load DQN and evaluate on an environment re-parameterized per contract.
    import torch
    from src.rl.env import AmericanPutEnv
    from src.rl.train_dqn import QNetwork, make_dqn_policy
    from src.rl.evaluate_policy import evaluate_policy

    online = QNetwork()
    online.load_state_dict(torch.load(args.dqn_checkpoint, map_location="cpu"))
    online.eval()
    dqn_policy = make_dqn_policy(online)

    def rl_eval_fn(contract: OptionContract) -> dict:
        env_factory = lambda seed: AmericanPutEnv(
            S0=contract.S0, K=contract.K, T=contract.T, r=contract.r,
            sigma=contract.sigma, steps=contract.steps, seed=seed,
        )
        return evaluate_policy(env_factory, dqn_policy, episodes=args.rl_episodes)

    run_comparison(nn_predict_fn, rl_eval_fn, out_path=args.out)
