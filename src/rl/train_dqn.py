"""
Train a DQN stopping policy (hold vs exercise) for the American put
environment. Cleaned up from Week 8: the original script mixed function
definitions with loose top-level training code (so importing it re-ran
training as a side effect). Here, training only runs under `if __name__ ==
"__main__"` / `main()`, so this module is safe to import from evaluation
and comparison scripts.

Usage:
    python -m src.rl.train_dqn --episodes 10000 --out reports/checkpoints/dqn.pth
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np

from src.rl.env import AmericanPutEnv

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for src.rl.train_dqn. Install with `pip install torch`."
    ) from e


class QNetwork(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def greedy_action(model, state) -> int:
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(model(state_t), dim=1).item())


def make_dqn_policy(model):
    return lambda state: greedy_action(model, state)


def compute_dqn_loss(online, target, batch, discount):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_selected = online(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        q_target = rewards + (1.0 - dones) * discount * next_q
    return F.smooth_l1_loss(q_selected, q_target)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_dqn(seed=42):
    set_seeds(seed)
    env = AmericanPutEnv(seed=seed)
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50_000)
    return env, online, target, optimizer, buffer


def train_dqn(env, online, target, optimizer, buffer, episodes=10_000, batch_size=128,
              target_update_every=250, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
              log_every=500):
    updates = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))

        while not done:
            if env.rng.random() < epsilon:
                action = int(env.rng.integers(0, 2))
            else:
                action = greedy_action(online, state)

            next_state, reward, done, info = env.step_env(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss = compute_dqn_loss(online, target, batch, env.discount)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                updates += 1
                if updates % target_update_every == 0:
                    target.load_state_dict(online.state_dict())

        if episode % log_every == 0:
            print(f"Episode {episode}/{episodes} | epsilon={epsilon:.4f} | updates={updates}")

    print("Training complete.")
    return online, target


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports/checkpoints/dqn.pth")
    args = parser.parse_args()

    env, online, target, optimizer, buffer = initialize_dqn(seed=args.seed)
    online, target = train_dqn(env, online, target, optimizer, buffer, episodes=args.episodes)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(online.state_dict(), args.out)
    print(f"Saved trained DQN to {args.out}")


if __name__ == "__main__":
    main()
