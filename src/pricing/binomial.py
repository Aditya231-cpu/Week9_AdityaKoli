"""
CRR binomial pricer for American (and European) options.

This is the anchor of the whole project: the neural network is trained to
approximate these prices, and the RL policy is judged against the exercise
boundary this model produces.

Adapted from the vectorized backward-induction routine used in Week 6
(crr_put_price_vectorized), extended here to also return the exercise
boundary (Deliverable: "Exercise boundary can be extracted or plotted").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .payoffs import payoff


@dataclass(frozen=True)
class OptionContract:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    steps: int = 100
    option_type: str = "put"

    def validate(self) -> None:
        assert self.S0 > 0, "S0 must be positive"
        assert self.K > 0, "K must be positive"
        assert self.T > 0, "T must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert self.steps >= 1, "steps must be >= 1"
        assert self.option_type in {"put", "call"}, "option_type must be put or call"


class PricingResult:
    def __init__(self, price: float, metadata: Optional[dict] = None):
        self.price = float(price)
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"PricingResult(price={self.price:.6f})"


class Pricer:
    name = "base"

    def price(self, contract: OptionContract) -> PricingResult:
        raise NotImplementedError


def crr_parameters(contract: OptionContract) -> Tuple[float, float, float, float, float]:
    """Compute CRR up/down/prob/discount/dt consistently, in one place."""
    dt = contract.T / contract.steps
    u = np.exp(contract.sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(contract.r * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is out of (0,1); "
            "check r, sigma, steps for arbitrage / stability issues."
        )
    df = np.exp(-contract.r * dt)
    return u, d, p, df, dt


def crr_american_put_with_boundary(contract: OptionContract) -> Tuple[float, Dict[int, float]]:
    """Backward-induction American option pricer that also records the
    exercise boundary (the lowest/highest spot at which exercise is
    optimal at each time step, for a put/call respectively).

    Returns
    -------
    price : float
        Time-0 option price.
    boundary : dict[int, float]
        Mapping step -> critical spot price at that step (the boundary
        between the continuation region and the exercise region).
    """
    contract.validate()
    u, d, p, df, dt = crr_parameters(contract)
    steps = contract.steps
    q = 1.0 - p
    is_american = True  # this pricer is specifically the American variant
    is_put = contract.option_type == "put"

    # terminal spot prices and payoffs
    S = contract.S0 * (d ** steps) * (u / d) ** np.arange(steps + 1)
    V = np.array([payoff(s, contract.K, contract.option_type) for s in S])

    boundary: Dict[int, float] = {}

    # record terminal boundary
    exercised = V > 0
    if exercised.any():
        boundary[steps] = float(S[exercised].max() if is_put else S[exercised].min())

    for i in range(steps - 1, -1, -1):
        S = S[:-1] * u
        continuation = df * (p * V[1:] + q * V[:-1])
        V = continuation
        if is_american:
            intrinsic = np.array([payoff(s, contract.K, contract.option_type) for s in S])
            exercise_mask = intrinsic > V
            V = np.maximum(V, intrinsic)
            if exercise_mask.any():
                spots = S[exercise_mask]
                boundary[i] = float(spots.max() if is_put else spots.min())

    return float(V[0]), boundary


def crr_price(contract: OptionContract, american: bool = True) -> float:
    """Convenience wrapper returning just the price (no boundary)."""
    if american:
        price, _ = crr_american_put_with_boundary(contract)
        return price

    # European fallback: plain backward induction, no early exercise check
    contract.validate()
    u, d, p, df, dt = crr_parameters(contract)
    steps = contract.steps
    q = 1.0 - p
    S = contract.S0 * (d ** steps) * (u / d) ** np.arange(steps + 1)
    V = np.array([payoff(s, contract.K, contract.option_type) for s in S])
    for _ in range(steps - 1, -1, -1):
        V = df * (p * V[1:] + q * V[:-1])
    return float(V[0])


class BinomialAmericanPutPricer(Pricer):
    name = "crr_binomial_american_put"

    def price(self, contract: OptionContract) -> PricingResult:
        price, boundary = crr_american_put_with_boundary(contract)
        return PricingResult(price, {"exercise_boundary": boundary})
