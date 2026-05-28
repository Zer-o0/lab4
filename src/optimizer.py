"""Configurable mini-batch SGD with optional momentum and adaptive rates.

Switches:
    batch_size: int (mini-batch size; 0 or None means full-batch)
    momentum: one of {"none", "classic", "nesterov"}
    adaptive: one of {"none", "adagrad", "rmsprop"}

Common presets:
    Plain SGD                : momentum="none",     adaptive="none"
    SGD + Momentum           : momentum="classic",  adaptive="none"
    SGD + Nesterov           : momentum="nesterov", adaptive="none"
    AdaGrad                  : momentum="none",     adaptive="adagrad"
    RMSProp                  : momentum="none",     adaptive="rmsprop"
    Adam (Momentum + RMSProp): momentum="classic",  adaptive="rmsprop"
    NAdam-like (Nesterov + RMSProp): momentum="nesterov", adaptive="rmsprop"

The "Nesterov look-ahead" requires the caller (model) to know how to evaluate
gradients at the look-ahead point. To keep the optimizer model-agnostic, we
implement Nesterov in the simplified Sutskever form:
    v_t = mu * v_{t-1} + lr * g_t
    theta_{t+1} = theta_t - (mu * v_t + lr * g_t)
which is equivalent to evaluating g at the look-ahead point under standard
training-loop conventions and is the standard PyTorch implementation.
"""
import numpy as np


_EPS = 1e-8


def iterate_minibatches(n_samples: int, batch_size: int, rng: np.random.Generator):
    """Yield arrays of indices for each mini-batch (shuffled per epoch)."""
    idx = rng.permutation(n_samples)
    if not batch_size or batch_size >= n_samples:
        yield idx
        return
    for start in range(0, n_samples, batch_size):
        yield idx[start : start + batch_size]


class SGD:
    def __init__(
        self,
        lr: float = 0.01,
        batch_size: int = 32,
        momentum: str = "none",
        beta1: float = 0.9,
        adaptive: str = "none",
        beta2: float = 0.999,
        bias_correction: bool = True,
    ):
        if momentum not in {"none", "classic", "nesterov"}:
            raise ValueError(f"Unknown momentum mode: {momentum}")
        if adaptive not in {"none", "adagrad", "rmsprop"}:
            raise ValueError(f"Unknown adaptive mode: {adaptive}")

        self.lr = float(lr)
        self.batch_size = int(batch_size) if batch_size else 0
        self.momentum = momentum
        self.beta1 = float(beta1)
        self.adaptive = adaptive
        self.beta2 = float(beta2)
        self.bias_correction = bool(bias_correction)

        self._v: dict[str, np.ndarray] = {}
        self._s: dict[str, np.ndarray] = {}
        self._t = 0

    def reset(self) -> None:
        self._v.clear()
        self._s.clear()
        self._t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """In-place parameter update."""
        self._t += 1
        for name, p in params.items():
            g = grads[name]

            if self.adaptive == "adagrad":
                s = self._s.setdefault(name, np.zeros_like(p))
                s += g * g
                g_eff = g / (np.sqrt(s) + _EPS)
            elif self.adaptive == "rmsprop":
                s = self._s.setdefault(name, np.zeros_like(p))
                s *= self.beta2
                s += (1.0 - self.beta2) * g * g
                s_hat = s / (1.0 - self.beta2 ** self._t) if self.bias_correction else s
                g_eff = g / (np.sqrt(s_hat) + _EPS)
            else:
                g_eff = g

            if self.momentum == "none":
                p -= self.lr * g_eff
            elif self.momentum == "classic":
                v = self._v.setdefault(name, np.zeros_like(p))
                v *= self.beta1
                v += (1.0 - self.beta1) * g_eff
                v_hat = v / (1.0 - self.beta1 ** self._t) if self.bias_correction else v
                p -= self.lr * v_hat
            else:  # nesterov (Sutskever form)
                v = self._v.setdefault(name, np.zeros_like(p))
                v *= self.beta1
                v += (1.0 - self.beta1) * g_eff
                v_hat = v / (1.0 - self.beta1 ** self._t) if self.bias_correction else v
                p -= self.lr * (self.beta1 * v_hat + (1.0 - self.beta1) * g_eff)

    @property
    def name(self) -> str:
        parts = []
        if self.momentum == "classic" and self.adaptive == "rmsprop":
            return "Adam"
        if self.momentum == "nesterov" and self.adaptive == "rmsprop":
            return "NAdam"
        if self.momentum == "classic":
            parts.append("Momentum")
        elif self.momentum == "nesterov":
            parts.append("Nesterov")
        if self.adaptive == "adagrad":
            parts.append("AdaGrad")
        elif self.adaptive == "rmsprop":
            parts.append("RMSProp")
        return "+".join(parts) if parts else "SGD"
