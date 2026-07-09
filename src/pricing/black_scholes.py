"""
Black-Scholes closed-form European option pricing.

Used only as a *reference point* in the final project. Black-Scholes does
not price American early exercise directly, so it is reported alongside
the CRR binomial benchmark to show the value of early-exercise optionality
(American price - European price >= 0 for puts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    d1: float
    d2: float


def black_scholes_price(S0: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = "put") -> BlackScholesResult:
    """Closed-form European Black-Scholes price.

    Parameters mirror OptionContract. Returns d1/d2 as well so callers can
    also report Greeks later if needed.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive for Black-Scholes.")

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type == "put":
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S0 * _norm_cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")

    return BlackScholesResult(price=float(price), d1=float(d1), d2=float(d2))
