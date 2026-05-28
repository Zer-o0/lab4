"""Models for binary classification, implemented from scratch on top of numpy.

Two architectures:
    SingleLayerNet: logistic regression  (Task 1)
        z  = X W + b
        y_hat = sigmoid(z)

    MLP1Hidden: one hidden layer with BReLU + sigmoid output  (Task 5)
        z1 = X W1 + b1
        h  = BReLU(z1)
        z2 = h W2 + b2
        y_hat = sigmoid(z2)

Both models expose:
    forward(X)  -> probabilities, shape (n, 1)
    backward(X, y_true) using cached pre-activations from the last forward
    params (dict[str, ndarray])
    fit(...) training loop with mini-batch SGD-family optimizer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from activations import BReLU, Sigmoid
from losses import BinaryCrossEntropy
from optimizer import SGD, iterate_minibatches


def _as_col(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1, 1).astype(np.float64)
    return y


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class _BaseClassifier:
    params: dict[str, np.ndarray]

    def forward(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X).ravel() >= threshold).astype(np.int64)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        optimizer: Optional[SGD] = None,
        epochs: int = 200,
        seed: int = 42,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = False,
    ) -> History:
        if optimizer is None:
            optimizer = SGD(lr=0.05, batch_size=32)
        optimizer.reset()
        y_train_col = _as_col(y_train)
        y_val_col = _as_col(y_val) if y_val is not None else None

        loss_fn = BinaryCrossEntropy()
        rng = np.random.default_rng(seed)
        history = History()

        best_val = np.inf
        best_params = {k: v.copy() for k, v in self.params.items()}
        epochs_since_improve = 0

        for epoch in range(epochs):
            for idx in iterate_minibatches(X_train.shape[0], optimizer.batch_size, rng):
                Xb = X_train[idx]
                yb = y_train_col[idx]
                _ = self.forward(Xb)
                grads = self._backward(Xb, yb)
                optimizer.step(self.params, grads)

            train_pred = self.predict_proba(X_train)
            train_loss = loss_fn(y_train_col, train_pred)
            train_acc = float(
                np.mean((train_pred.ravel() >= 0.5).astype(np.int64) == y_train)
            )
            history.train_loss.append(train_loss)
            history.train_acc.append(train_acc)

            if X_val is not None and y_val_col is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = loss_fn(y_val_col, val_pred)
                val_acc = float(
                    np.mean((val_pred.ravel() >= 0.5).astype(np.int64) == y_val)
                )
                history.val_loss.append(val_loss)
                history.val_acc.append(val_acc)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_params = {k: v.copy() for k, v in self.params.items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if (
                        early_stopping_patience is not None
                        and epochs_since_improve >= early_stopping_patience
                    ):
                        if verbose:
                            print(f"Early stop at epoch {epoch}, best val_loss={best_val:.4f}")
                        break

            if verbose and (epoch % max(1, epochs // 10) == 0):
                msg = f"epoch {epoch:4d}  train_loss={train_loss:.4f} acc={train_acc:.3f}"
                if X_val is not None:
                    msg += f"  val_loss={history.val_loss[-1]:.4f} acc={history.val_acc[-1]:.3f}"
                print(msg)

        if X_val is not None and y_val_col is not None:
            self.params = best_params  # restore best
        return history


class SingleLayerNet(_BaseClassifier):
    """Single-layer perceptron: y = sigmoid(X W + b).

    Task 1's "single-link" network. BReLU is not present in this architecture
    (it acts as identity over the activated region of a sigmoid output and
    only restricts the predictable range, see the report); the BReLU is
    instead used in Task 5's hidden layer.
    """

    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Glorot-ish init for sigmoid output layer
        limit = np.sqrt(6.0 / (n_features + 1))
        self.params = {
            "W": rng.uniform(-limit, limit, size=(n_features, 1)),
            "b": np.zeros((1, 1)),
        }
        self._sigmoid = Sigmoid()
        self._cache: dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.params["W"] + self.params["b"]
        y_hat = self._sigmoid(z)
        self._cache = {"z": z, "y_hat": y_hat}
        return y_hat

    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        y_hat = self._cache["y_hat"]
        n = X.shape[0]
        # combined dL/dz for sigmoid + BCE
        dz = (y_hat - y_true) / n
        dW = X.T @ dz
        db = dz.sum(axis=0, keepdims=True)
        return {"W": dW, "b": db}


class MLP1Hidden(_BaseClassifier):
    """One hidden layer with BReLU; sigmoid output.

    Architecture:  X (n,d) -> Linear(d,H) -> BReLU -> Linear(H,1) -> Sigmoid
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 16,
        brelu_bound: float = 1.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        # He init for the BReLU hidden layer
        he = np.sqrt(2.0 / n_features)
        # Glorot for the sigmoid output layer
        glorot = np.sqrt(6.0 / (hidden_size + 1))

        self.params = {
            "W1": rng.normal(0.0, he, size=(n_features, hidden_size)),
            "b1": np.zeros((1, hidden_size)),
            "W2": rng.uniform(-glorot, glorot, size=(hidden_size, 1)),
            "b2": np.zeros((1, 1)),
        }
        self._sigmoid = Sigmoid()
        self._brelu = BReLU(bound=brelu_bound)
        self._cache: dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.params["W1"] + self.params["b1"]
        h = self._brelu(z1)
        z2 = h @ self.params["W2"] + self.params["b2"]
        y_hat = self._sigmoid(z2)
        self._cache = {"z1": z1, "h": h, "z2": z2, "y_hat": y_hat}
        return y_hat

    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, np.ndarray]:
        z1 = self._cache["z1"]
        h = self._cache["h"]
        y_hat = self._cache["y_hat"]
        n = X.shape[0]

        dz2 = (y_hat - y_true) / n
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        dh = dz2 @ self.params["W2"].T
        dz1 = dh * self._brelu.backward(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


if __name__ == "__main__":
    # Smoke test: fit on moons; verify both architectures train.
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    from utils import seed_everything, split_train_val_test

    seed_everything(0)
    X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
    split = split_train_val_test(X, y, random_state=0)
    sc = StandardScaler().fit(split.X_train)
    Xtr, Xva, Xte = sc.transform(split.X_train), sc.transform(split.X_val), sc.transform(split.X_test)

    print("--- SingleLayerNet ---")
    slp = SingleLayerNet(n_features=Xtr.shape[1], seed=0)
    opt = SGD(lr=0.1, batch_size=32, momentum="classic", adaptive="rmsprop")  # Adam
    h = slp.fit(Xtr, split.y_train, Xva, split.y_val, optimizer=opt, epochs=200, seed=0)
    print(f"Best val_loss={min(h.val_loss):.4f}, test_acc={np.mean(slp.predict(Xte) == split.y_test):.3f}")

    print("--- MLP1Hidden ---")
    mlp = MLP1Hidden(n_features=Xtr.shape[1], hidden_size=16, brelu_bound=1.0, seed=0)
    opt = SGD(lr=0.1, batch_size=32, momentum="classic", adaptive="rmsprop")  # Adam
    h = mlp.fit(Xtr, split.y_train, Xva, split.y_val, optimizer=opt, epochs=300, seed=0)
    print(f"Best val_loss={min(h.val_loss):.4f}, test_acc={np.mean(mlp.predict(Xte) == split.y_test):.3f}")
