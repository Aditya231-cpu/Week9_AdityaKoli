"""
Evaluate RL (and baseline) stopping policies by Monte-Carlo simulation,
and compare the learned exercise region against the CRR binomial boundary.
"""
from __future__ import annotations

import random

import numpy as np

from src.rl.env import AmericanPutEnv


def evaluate_policy(env_factory, policy_fn, episodes: int = 5000, base_seed: int = 10_000) -> dict:
    """Estimate policy value via Monte Carlo on out-of-sample seeds.

    Returns value, standard error, exercise rate, and average exercise step
    -- the full set the project spec asks for.
    """
    discounted_rewards = []
    exercise_steps = []

    for i in range(episodes):
        env = env_factory(seed=base_seed + i)
        state = env.reset()
        done = False
        step = 0
        while not done:
            action = policy_fn(state)
            state, reward, done, info = env.step_env(action)
            if done:
                discounted_rewards.append((env.discount ** step) * reward)
                if info["reason"] == "exercise":
                    exercise_steps.append(step)
            step += 1

    discounted_rewards = np.array(discounted_rewards)
    return {
        "value": float(np.mean(discounted_rewards)),
        "std_error": float(np.std(discounted_rewards) / np.sqrt(episodes)),
        "exercise_rate": len(exercise_steps) / episodes,
        "avg_exercise_step": float(np.mean(exercise_steps)) if exercise_steps else None,
    }


# ---- Baseline policies (Deliverable: compare RL vs these) ------------------

def always_hold_policy(state):
    return AmericanPutEnv.HOLD


def immediate_exercise_policy(state):
    return AmericanPutEnv.EXERCISE if state[0] == 0 else AmericanPutEnv.HOLD


def random_policy(state):
    return random.choice([AmericanPutEnv.HOLD, AmericanPutEnv.EXERCISE])


def build_policy_comparison(env_factory, policies: dict, episodes: int = 10_000):
    import pandas as pd
    rows = []
    for name, policy_fn in policies.items():
        result = evaluate_policy(env_factory, policy_fn, episodes=episodes)
        rows.append({"policy": name, **result})
    return pd.DataFrame(rows).sort_values("value", ascending=False)


def policy_grid(policy_fn, steps=50, money_min=0.7, money_max=1.3, n_money=13):
    """Text-friendly hold/exercise grid, sampled every 5 steps."""
    grid = []
    moneyness_values = np.linspace(money_min, money_max, n_money)
    for step in range(0, steps + 1, 5):
        row = []
        for m in moneyness_values:
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            row.append("X" if policy_fn(state) == 1 else ".")
        grid.append((step, row))
    return grid, moneyness_values


def boundary_agreement(policy_fn, boundary_by_step, steps=100, K=100.0):
    """Fraction of (step, moneyness) grid points where the RL policy agrees
    with the binomial exercise boundary. Requires the policy's state
    format [time_fraction, time_to_expiry, moneyness]."""
    checks = []
    money_grid = np.linspace(0.6, 1.4, 81)
    for step in range(steps):
        boundary_spot = boundary_by_step.get(step)
        if boundary_spot is None:
            continue
        for m in money_grid:
            S = m * K
            state = np.array([step / steps, 1.0 - step / steps, m], dtype=np.float32)
            policy_exercise = policy_fn(state) == 1
            binomial_exercise = S <= boundary_spot
            checks.append(policy_exercise == binomial_exercise)
    return float(np.mean(checks)) if checks else None
