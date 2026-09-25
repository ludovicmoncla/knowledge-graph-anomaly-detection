from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path = Path("outputs/serad_kg")
    prepared_data_dir: Path | None = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    compressed_dim: int = 64
    plausibility_weight: float = 0.5
    auxiliary_loss_weight: float = 0.25
    ranking_loss_weight: float = 0.0
    ranking_margin: float = 0.5
    score_normalization: str = "dynamic"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 25
    min_delta: float = 1e-5
    edge_mask_ratio: float = 0.0
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    random_seed: int = 42
    test_size: float = 0.2
    # 25% of the remaining 80% yields the same global 60/20/20 split as LoGNet.
    validation_size: float = 0.25
    anomaly_count: int = 30
    negative_sampling: str = "random"
    anomaly_cache_dir: Path | None = None
    openrouter_model: str = "openai/gpt-4.1"
    env_file: Path = Path(".env")
    max_snapshots: int | None = 10
    min_triples: int = 3
    device: str = "auto"
    save_model: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.plausibility_weight <= 1.0:
            raise ValueError("plausibility_weight must be between 0 and 1")
        if self.auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative")
        if self.ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be non-negative")
        if self.ranking_margin < 0.0:
            raise ValueError("ranking_margin must be non-negative")
        if self.score_normalization not in ("stable", "dynamic"):
            raise ValueError("score_normalization must be 'stable' or 'dynamic'")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if not 0.0 <= self.edge_mask_ratio < 1.0:
            raise ValueError("edge_mask_ratio must be between 0 (inclusive) and 1 (exclusive)")
        if self.scheduler_patience < 1:
            raise ValueError("scheduler_patience must be at least 1")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be between 0 and 1")
        if self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be positive")

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
