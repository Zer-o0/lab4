"""Utilities: seeding, train/val/test split, classification metrics."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Split:
    """Stratified 60/20/20 split via two train_test_split calls."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_share = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_share,
        stratify=y_trainval,
        random_state=random_state,
    )
    return Split(X_train, y_train, X_val, y_val, X_test, y_test)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }
