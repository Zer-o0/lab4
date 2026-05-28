"""Активации.

Sigmoid используется в выходном слое: производная сворачивается с производной
функции потерь (BCE) в выражение (y_hat - y) / n, поэтому отдельный
sigmoid.backward в модели не нужен.

BReLU(B) = min(max(z, 0), B). Производная равна 1 строго внутри (0, B) и 0
вне — она нужна для обратного прохода в скрытом слое.
"""
import numpy as np


class Sigmoid:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out


class BReLU:
    def __init__(self, bound: float = 1.0):
        self.bound = float(bound)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.clip(z, 0.0, self.bound)

    def backward(self, z: np.ndarray) -> np.ndarray:
        return ((z > 0.0) & (z < self.bound)).astype(np.float64)
