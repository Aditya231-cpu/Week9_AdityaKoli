"""
Option payoff functions.

These are the terminal / exercise payoffs used by every pricer and by the
RL environment reward function, so they live in one place and are unit
tested directly (see tests/test_payoffs.py).
"""
from __future__ import annotations


def put_payoff(S: float, K: float) -> float:
    """Intrinsic value of a put: max(K - S, 0)."""
    return max(K - S, 0.0)


def call_payoff(S: float, K: float) -> float:
    """Intrinsic value of a call: max(S - K, 0)."""
    return max(S - K, 0.0)


def payoff(S: float, K: float, option_type: str = "put") -> float:
    """Dispatch to the correct payoff function based on option_type."""
    if option_type == "put":
        return put_payoff(S, K)
    if option_type == "call":
        return call_payoff(S, K)
    raise ValueError(f"Unknown option_type: {option_type!r}")
