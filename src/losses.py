"""Бинарная перекрёстная энтропия.

L = -среднее(y*log(p) + (1-y)*log(1-p)).
Перед log выполняется ограничение p в [eps, 1-eps], чтобы избежать NaN.

Производная по выходу сигмоиды свёртывается аналитически:
    dL/dz = (y_hat - y) / n,
поэтому отдельный метод backward здесь не нужен — он реализован прямо в
моделях.
"""
import numpy as np

_EPS = 1e-12


class BinaryCrossEntropy:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
