from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def roc_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    index = np.argmin(np.hypot(false_positive_rate, 1 - true_positive_rate))
    return float(thresholds[index]), float(auc(false_positive_rate, true_positive_rate))


def save_loss_plot(train_losses: list[float], validation_losses: list[float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(train_losses, label="Train")
    axis.plot(validation_losses, label="Validation", linestyle="--")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="Training history")
    axis.grid(linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_roc_plot(labels: np.ndarray, scores: np.ndarray, title: str, path: Path) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    area = float(auc(false_positive_rate, true_positive_rate))
    figure, axis = plt.subplots()
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {area:.3f}")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return area
