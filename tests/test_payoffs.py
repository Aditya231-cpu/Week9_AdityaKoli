import pytest

from src.pricing.payoffs import put_payoff, call_payoff, payoff


def test_put_payoff_itm():
    assert put_payoff(80.0, 100.0) == 20.0


def test_put_payoff_otm_is_zero():
    assert put_payoff(120.0, 100.0) == 0.0


def test_call_payoff_itm():
    assert call_payoff(120.0, 100.0) == 20.0


def test_call_payoff_otm_is_zero():
    assert call_payoff(80.0, 100.0) == 0.0


def test_payoff_never_negative():
    for S in [0.0, 50.0, 100.0, 150.0, 300.0]:
        assert put_payoff(S, 100.0) >= 0.0
        assert call_payoff(S, 100.0) >= 0.0


def test_payoff_dispatch_matches_direct_call():
    assert payoff(90.0, 100.0, "put") == put_payoff(90.0, 100.0)
    assert payoff(110.0, 100.0, "call") == call_payoff(110.0, 100.0)


def test_payoff_invalid_type_raises():
    with pytest.raises(ValueError):
        payoff(100.0, 100.0, "straddle")
