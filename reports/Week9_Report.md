# American Option Pricing Capstone — Final Report

**Author:** _Your Name_
**Course:** Weeks 1–9 Final Project
**Repository:** _add your GitHub link here_

> **How to use this template:** the code, tests, and figure-generation
> pipeline in this repository are complete and tested. The tables/figures
> below marked `[[RUN PIPELINE]]` are placeholders — run the commands in
> the README (`train_nn.py`, `train_dqn.py`, `comparison.py`, `plots.py`)
> with your chosen episode/epoch counts, then paste your actual numbers
> and swap in the generated PNGs before exporting this file to PDF.

---

## 1. Executive summary

This project prices American put options three ways — a CRR binomial
tree (benchmark), a neural network trained to approximate binomial
prices, and a reinforcement-learning (DQN) policy that learns exercise
decisions directly — and compares them on one shared contract grid.

`[[RUN PIPELINE]]` One-paragraph summary once results are in, e.g.:
*"The NN pricer matches the binomial benchmark closely near-the-money
(MAE ≈ X) but degrades in deep OTM regions where relative error is
unstable. The DQN policy recovers an exercise boundary that is
qualitatively consistent with the binomial boundary (Y% agreement) and
outperforms the always-hold and random baselines, though it slightly
under-exercises relative to the binomial-optimal boundary at short
maturities."*

## 2. Problem statement: American put pricing and early exercise

An American put gives the holder the right to sell an asset at strike
`K` at or before maturity `T`. Unlike a European option, it can be
exercised at any time, which means its price is not a simple discounted
expectation — it is the solution to an optimal-stopping problem: at
every point in time, the holder compares the immediate exercise value
`max(K - S, 0)` to the value of continuing to hold. This early-exercise
feature is what makes American puts harder to price than European ones,
and it is the central object this project studies with three different
tools.

## 3. Financial background

- **Payoff.** `src/pricing/payoffs.py` — `put_payoff(S, K) = max(K - S, 0)`.
- **Black-Scholes context.** `src/pricing/black_scholes.py` gives the
  closed-form European price. It is a useful reference point (and a
  lower bound for the American price, since American ≥ European), but it
  does not itself solve the optimal-stopping problem, which is why the
  project needs the binomial tree, not just Black-Scholes.
- **Binomial model.** `src/pricing/binomial.py` discretizes the
  risk-neutral stock process into a CRR lattice with up/down factors `u,
  d` and risk-neutral probability `p`, then solves the optimal-stopping
  problem by backward induction: `V(t,S) = max(K - S, exp(-r·dt) ·
  E[V(t+dt, S_next)])`.

## 4. Method 1: CRR binomial benchmark

`crr_american_put_with_boundary()` computes `u`, `d`, `p` consistently
from `(r, sigma, dt)`, then runs backward induction, taking the
elementwise max of continuation value and intrinsic value at every node.
It also records, at each time step, the critical spot price separating
the hold region from the exercise region — the **exercise boundary**.

This is the anchor of the whole project: NN training labels and RL
evaluation both reference this pricer.

**Figure 1 — payoff diagram:**
`reports/figures/01_payoff_diagram.png`

**Figure 2 — binomial exercise boundary** (K=100, T=1, r=5%, σ=25%):
`reports/figures/02_binomial_exercise_boundary.png`

The boundary curve rises toward maturity — near expiry, exercise is
optimal at spot prices close to the strike, whereas far from expiry the
option is worth more alive (holding preserves optionality), so the
critical exercise spot is lower. This shape is the standard qualitative
signature of an American put boundary and is a first sanity check that
the tree implementation is correct.

## 5. Method 2: neural-network price approximation

`src/ml/models.py` implements a small MLP (5 → 256 → 256 → 1, ReLU,
Adam) trained on `(S0, K, T, r, sigma) → binomial_price` pairs generated
by `src/data/synthetic_contracts.make_training_dataset()`. Inputs are
standardized using training-set statistics only. Training/validation/test
split is 80/10/10; the checkpoint with lowest validation MSE is kept
(`src/ml/train_nn.py`).

**Why this is a regression problem, and why that's not the whole
story:** accuracy metrics alone (MAE/RMSE) can hide financially
nonsensical behavior. `src/ml/evaluate_nn.py` additionally checks:

- Intrinsic-value bound: NN price should never fall below `max(K-S, 0)`.
- Spot monotonicity: put price should not rise as spot rises.
- Extrapolation flag: whether a queried contract falls outside the
  training ranges in `configs/nn.yaml`.

`[[RUN PIPELINE]]` Fill in after running `train_nn.py` + `evaluate_nn.py`:

| Metric | Overall | Deep ITM put | ATM | Deep OTM put |
|---|---|---|---|---|
| MAE | | | | |
| RMSE | | | | |
| Max abs error | | | | |
| Mean bias | | | | |
| Intrinsic-value violations (count) | | | | |
| Monotonicity violations (count) | | | | |

## 6. Method 3: RL stopping policy

`src/rl/env.py` implements `AmericanPutEnv`, a risk-neutral
geometric-Brownian-motion simulator where, at each step, the agent
chooses `HOLD` (advance one step) or `EXERCISE` (terminate and collect
`max(K - S, 0)`). State is `[time_fraction, time_to_expiry, moneyness]`.
`src/rl/train_dqn.py` trains a DQN (two-layer Q-network, replay buffer,
target network, epsilon-greedy exploration) to maximize discounted
reward — i.e. to learn the optimal-stopping policy directly from
simulation, without ever seeing a binomial price.

A second environment, `AmericanPutLatticeEnv`, steps the CRR lattice
directly (as in the Week 7 tabular Q-learning version) and is kept for
cross-checking the DQN's learned region against a tabular baseline
trained on the same discretization the binomial pricer uses.

## 7. Experimental setup: contract grid, parameters, seeds, metrics

**Shared evaluation grid** (`src/data/synthetic_contracts.make_contract_grid`):
spots `{70,...,130}` (step 10), strike fixed at 100, maturities
`{0.25, 0.5, 1.0, 2.0}`, rates `{0.02, 0.05}`, volatilities
`{0.15, 0.25, 0.40}`, 100 binomial steps — 168 contracts spanning ITM,
ATM, and OTM regions, short and long maturities, low and high
volatility.

**Seeds:** `seed=42` throughout (data generation, NN init/training
shuffle, DQN init/exploration/replay sampling) — see `configs/nn.yaml`,
`configs/rl.yaml`.

**Metrics:** NN — MAE, RMSE, max absolute error, mean bias, median
absolute error, mean relative error (`src/evaluation/metrics.py`). RL —
Monte Carlo policy value, standard error, exercise rate, average
exercise step (`src/rl/evaluate_policy.py`), plus boundary-agreement
rate against the binomial boundary.

## 8. Results: tables and plots

`[[RUN PIPELINE]]` — regenerate and paste in:

- `reports/figures/03_nn_vs_binomial_scatter.png` — NN predicted vs
  binomial benchmark price, all 168 grid contracts.
- `reports/figures/04_nn_error_heatmap.png` — mean NN error by
  maturity × volatility.
- `reports/figures/05_rl_exercise_region.png` — DQN learned hold/exercise
  region over time and moneyness.
- `reports/comparison.csv` / `reports/comparison_by_bucket.csv` — the
  unified table (binomial price, NN price, NN error, RL value, RL
  standard error, RL exercise rate) and its ITM/ATM/OTM summary.

**Policy comparison table** (fill in from `build_policy_comparison`):

| Policy | Value | Std error | Exercise rate | Avg exercise step |
|---|---|---|---|---|
| Always hold | | | | |
| Immediate exercise | | | | |
| Random | | | | |
| Trained DQN | | | | |
| Binomial (reference) | | — | — | — |

## 9. Discussion: strengths, failures, model risk, limitations

`[[RUN PIPELINE]]` Discuss, once results are in:
- Where does the NN over/underprice (check `mean_bias` by bucket)?
- Does the DQN's exercise region match the binomial boundary shape
  (rising toward expiry, bounded near the strike)?
- Which method is fastest at inference after training (binomial:
  O(steps²) per contract; NN: one forward pass; DQN: one forward pass
  per decision, but requires simulating a full path to get a policy
  value)?
- Which is easiest to trust and explain? (Binomial: fully transparent.
  NN: accurate but a black box, needs the sanity checks above. RL:
  learns a good policy but Monte Carlo evaluation is noisy and the
  policy was trained on one fixed contract's dynamics — see below.)

**Model risk note — DQN generalization.** The DQN in this repo is
trained on one fixed contract (`S0=100, K=100, T=1, r=5%, σ=20%`,
50 steps; see `configs/rl.yaml`). `src/evaluation/comparison.py`
re-parameterizes the environment per grid contract at evaluation time,
which tests whether the learned policy generalizes across `T, r, sigma`
it never trained on — a stronger and more honest test than evaluating
only at its training settings, but also one where noticeably worse RL
performance away from `(T=1, r=0.05, sigma=0.20)` should be *expected*
and is not a bug. A production-grade version would train (or condition)
the policy across the same contract distribution used for NN training.

## 10. Reproducibility

- Repo link: _add here_
- Commands: see `README.md` → "Reproduce results"
- Environment: `requirements.txt` (Python 3.10+, NumPy, pandas,
  matplotlib, seaborn, SciPy, PyTorch, pytest)
- Seeds: `configs/nn.yaml`, `configs/rl.yaml` (`seed: 42`)
- Tests: `python -m pytest` — 22 tests covering payoffs, binomial pricing
  sanity, and RL environment invariants, all passing.

## 11. Conclusion and future improvements

`[[RUN PIPELINE]]` One paragraph tying together sections 8–9. Suggested
future work to mention: train the NN and DQN on matched contract
distributions so RL/NN comparisons are apples-to-apples; add dividends
and stochastic volatility; replace tabular DQN evaluation with a
variance-reduced estimator (e.g. control variates using the binomial
price) to shrink RL standard errors; calibrate to real market data
instead of only synthetic labels.

---

## Limitations

This project uses synthetic contracts and a CRR binomial benchmark
under simplified assumptions (no dividends, no transaction costs, no
stochastic volatility, no liquidity constraints, GBM/CRR risk-neutral
dynamics only). The neural network approximates the binomial benchmark
on the sampled grid, so it cannot exceed benchmark truth by construction
— any "accuracy" is accuracy relative to the tree, not to real markets.
The RL policy is evaluated by Monte Carlo simulation in the same
stylized environment, and its training was noisy and dependent on
reward design, seeds, and exploration schedule. No real market
calibration was performed. Predictions or policy decisions outside the
sampled training/evaluation ranges should not be trusted without
further validation. Results should be read as a learning and
prototyping exercise, not as production trading advice or a calibrated
market-pricing system.
