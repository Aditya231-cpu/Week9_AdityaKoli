"""
Neural-network pricer model.

This is the Week 6 `NumPyAdamMLP` (a from-scratch MLP trained with Adam),
kept dependency-light (NumPy only) so the whole capstone can run without
requiring a deep-learning framework just for the pricer. The RL side
(src/rl) uses PyTorch separately for the DQN.
"""
from __future__ import annotations

import numpy as np


class NumPyAdamMLP:
    """A small MLP (5 -> hidden -> hidden -> 1) optimized with Adam.

    Input features are [S0, K, T, r, sigma] (standardized before entering
    the network); output is the predicted American put price.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 256, output_dim: int = 1, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He-normal init for ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim))
        self.b2 = np.zeros((1, hidden_dim))
        # Glorot init for the linear output layer
        self.W3 = rng.normal(0, np.sqrt(1.0 / hidden_dim), (hidden_dim, output_dim))
        self.b3 = np.zeros((1, output_dim))

        self.m_W1, self.v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.m_b1, self.v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.m_W2, self.v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.m_b2, self.v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.m_W3, self.v_W3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.m_b3, self.v_b3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        self.t = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        m = y_true.shape[0]
        dy = (y_pred - y_true) / m

        self.dW3 = self.a2.T @ dy
        self.db3 = np.sum(dy, axis=0, keepdims=True)

        da2 = dy @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        self.dW1 = self.X.T @ dz1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.t += 1
        for param, grad, m, v in [
            (self.W1, self.dW1, self.m_W1, self.v_W1),
            (self.b1, self.db1, self.m_b1, self.v_b1),
            (self.W2, self.dW2, self.m_W2, self.v_W2),
            (self.b2, self.db2, self.m_b2, self.v_b2),
            (self.W3, self.dW3, self.m_W3, self.v_W3),
            (self.b3, self.db3, self.m_b3, self.v_b3),
        ]:
            m *= beta1
            m += (1 - beta1) * grad
            v *= beta2
            v += (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def get_state(self) -> dict:
        return {k: getattr(self, k).copy() for k in ("W1", "b1", "W2", "b2", "W3", "b3")}

    def set_state(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v.copy())
