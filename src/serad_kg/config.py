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
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    epochs: int = 150
    patience: int = 15
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

    def __post_init__(self) -> None:
        if not 0.0 <= self.plausibility_weight <= 1.0:
            raise ValueError("plausibility_weight must be between 0 and 1")

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
