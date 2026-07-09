import numpy as np
import pytest

from src.rl.env import AmericanPutEnv, AmericanPutLatticeEnv, discretize_state


def test_reward_never_negative_gbm_env():
    env = AmericanPutEnv(seed=101)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.reset()
        done = False
        while not done:
            action = int(rng.integers(0, 2))
            _, reward, done, _ = env.step_env(action)
            assert reward >= 0.0


def test_cannot_step_after_done_gbm_env():
    env = AmericanPutEnv(seed=102)
    env.reset()
    _, _, done, _ = env.step_env(AmericanPutEnv.EXERCISE)
    assert done
    with pytest.raises(RuntimeError):
        env.step_env(AmericanPutEnv.HOLD)


def test_episode_terminates_by_expiry_if_never_exercised():
    env = AmericanPutEnv(steps=20, seed=5)
    env.reset()
    done = False
    n_steps = 0
    while not done:
        _, _, done, info = env.step_env(AmericanPutEnv.HOLD)
        n_steps += 1
        assert n_steps <= 20  # must terminate at expiry, not run forever
    assert info["reason"] == "expiry"


def test_state_shape_and_bounds():
    env = AmericanPutEnv(seed=1)
    state = env.reset()
    assert state.shape == (3,)
    assert 0.0 <= state[0] <= 1.0  # time fraction
    assert state[2] > 0  # moneyness positive


def test_lattice_env_reward_never_negative():
    env = AmericanPutLatticeEnv(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.reset()
        done = False
        while not done:
            action = int(rng.integers(0, 2))
            _, reward, done, _ = env.step_env(action)
            assert reward >= 0.0


def test_invalid_action_raises():
    env = AmericanPutEnv(seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step_env(99)


def test_discretize_state_within_bounds():
    for state in [(0.0, 0.5), (1.0, 1.5), (0.5, 1.0), (0.99, 0.4)]:
        idx = discretize_state(np.array(state), n_time=20, n_money=30)
        assert 0 <= idx[0] < 20
        assert 0 <= idx[1] < 30
