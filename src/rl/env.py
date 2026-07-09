"""
American put exercise environment (hold vs exercise) for RL.

This consolidates the two environment variants written across Weeks 7-8
into one canonical version:
  - Week 7's env stepped the *risk-neutral CRR lattice* (u/d/p).
  - Week 8's env stepped *risk-neutral GBM* directly (continuous spot).

We keep the GBM version as canonical because it is the one the DQN was
trained against in Week 8 and it does not depend on a discretization
choice (steps), which makes it a fairer, tree-independent test of the RL
policy against the binomial *boundary* (comparing a discrete-tree method
to a policy trained on a different discretization would confound the
comparison). The CRR-lattice variant is kept as `AmericanPutLatticeEnv`
below for completeness / consistency checks against Week 7 results.
"""
from __future__ import annotations

import math

import numpy as np


class AmericanPutEnv:
    """Risk-neutral GBM environment. State = [time_fraction, time_to_expiry, moneyness]."""

    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, steps=50, seed=42):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.dt = T / steps
        self.discount = math.exp(-self.r * self.dt)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def make_state(self, step, spot):
        time_fraction = step / self.steps
        time_to_expiry = 1.0 - time_fraction
        moneyness = spot / self.K
        return np.array([time_fraction, time_to_expiry, moneyness], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.current_spot = self.S0
        self.done = False
        return self.make_state(self.current_step, self.current_spot)

    def step_env(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")
        if action not in (self.HOLD, self.EXERCISE):
            raise ValueError("action must be 0=hold or 1=exercise")

        if action == self.EXERCISE:
            reward = max(0.0, self.K - self.current_spot)
            self.done = True
            return self.make_state(self.current_step, self.current_spot), reward, True, {"reason": "exercise"}

        self.current_step += 1
        Z = self.rng.standard_normal()
        drift = (self.r - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * Z
        self.current_spot *= math.exp(drift + diffusion)

        next_state = self.make_state(self.current_step, self.current_spot)

        if self.current_step >= self.steps:
            reward = max(0.0, self.K - self.current_spot)
            self.done = True
            return next_state, reward, True, {"reason": "expiry"}

        return next_state, 0.0, False, {"reason": "hold"}


class AmericanPutLatticeEnv:
    """CRR-lattice environment (Week 7 variant). State = [time_fraction, moneyness].

    Kept for cross-checking the DQN's learned boundary against a tabular
    Q-learning policy trained directly on the same lattice the binomial
    pricer uses.
    """

    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50, seed=42):
        self.S0, self.K, self.T, self.r, self.sigma, self.steps = S0, K, T, r, sigma, steps
        self.dt = T / steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _state(self):
        return np.array([self.step / self.steps, self.spot / self.K], dtype=np.float32)

    def reset(self):
        self.step = 0
        self.spot = self.S0
        self.done = False
        return self._state()

    def step_env(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")
        if action not in (self.HOLD, self.EXERCISE):
            raise ValueError("action must be 0=hold or 1=exercise")

        payoff = max(self.K - self.spot, 0.0)
        if action == self.EXERCISE:
            self.done = True
            return self._state(), payoff, True, {"reason": "exercise"}

        self.spot *= self.u if self.rng.random() < self.p else self.d
        self.step += 1

        if self.step >= self.steps:
            self.done = True
            terminal_payoff = max(self.K - self.spot, 0.0)
            return self._state(), terminal_payoff, True, {"reason": "expiry"}

        return self._state(), 0.0, False, {"reason": "hold"}


def discretize_state(state, n_time=20, n_money=30, money_min=0.5, money_max=1.5):
    """Bucketize a continuous (time_fraction, moneyness) state for tabular Q-learning."""
    t_idx = min(int(state[0] * n_time), n_time - 1)
    m_clamped = min(max(state[1], money_min), money_max - 1e-9)
    m_idx = int((m_clamped - money_min) / (money_max - money_min) * n_money)
    m_idx = min(max(m_idx, 0), n_money - 1)
    return (t_idx, m_idx)
