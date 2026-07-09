import pytest

from src.pricing.binomial import OptionContract, crr_price, crr_american_put_with_boundary


def test_zero_volatility_put_matches_discounted_intrinsic():
    # With sigma -> 0 the tree degenerates; use a tiny sigma and confirm the
    # price is close to the (deterministic-ish) discounted intrinsic value
    # when the option is deep ITM and r is small.
    contract = OptionContract(S0=50.0, K=100.0, T=1.0, r=0.0, sigma=0.05, steps=200, option_type="put")
    price = crr_price(contract, american=True)
    intrinsic = max(contract.K - contract.S0, 0.0)
    assert price == pytest.approx(intrinsic, abs=1.0)


def test_price_never_below_intrinsic():
    contract = OptionContract(S0=90.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=200, option_type="put")
    price = crr_price(contract, american=True)
    intrinsic = max(contract.K - contract.S0, 0.0)
    assert price >= intrinsic - 1e-8


def test_american_put_at_least_european_put():
    contract = OptionContract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=200, option_type="put")
    american = crr_price(contract, american=True)
    european = crr_price(contract, american=False)
    assert american >= european - 1e-8


def test_put_price_decreases_with_spot():
    K, T, r, sigma, steps = 100.0, 1.0, 0.05, 0.25, 150
    prices = []
    for S0 in [70, 90, 110, 130]:
        c = OptionContract(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=steps, option_type="put")
        prices.append(crr_price(c, american=True))
    assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))


def test_put_price_increases_with_strike():
    S0, T, r, sigma, steps = 100.0, 1.0, 0.05, 0.25, 150
    prices = []
    for K in [80, 90, 100, 110]:
        c = OptionContract(S0=S0, K=K, T=T, r=r, sigma=sigma, steps=steps, option_type="put")
        prices.append(crr_price(c, american=True))
    assert all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))


def test_exercise_boundary_is_nonempty_for_itm_regime():
    contract = OptionContract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=100, option_type="put")
    price, boundary = crr_american_put_with_boundary(contract)
    assert price > 0
    assert len(boundary) > 0
    # boundary spots should be below strike for a put
    assert all(spot <= contract.K for spot in boundary.values())


def test_contract_validate_rejects_bad_inputs():
    with pytest.raises(AssertionError):
        OptionContract(S0=-1.0, K=100.0, T=1.0, r=0.05, sigma=0.25).validate()
    with pytest.raises(AssertionError):
        OptionContract(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, option_type="bad").validate()


def test_invalid_p_raises_value_error():
    # steps=1 with extreme sigma/r can push p outside (0,1)
    contract = OptionContract(S0=100.0, K=100.0, T=5.0, r=2.0, sigma=0.01, steps=1, option_type="put")
    with pytest.raises(ValueError):
        crr_price(contract, american=True)
