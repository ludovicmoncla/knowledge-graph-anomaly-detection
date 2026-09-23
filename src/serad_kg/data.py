from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


@dataclass
class Vocabulary:
    text_to_id: dict[str, int]
    id_to_text: dict[int, str]


@dataclass
class PreparedSnapshot:
    graph: Data
    triples: torch.Tensor
    triples_numpy: np.ndarray
    labels: torch.Tensor
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    test_mask: torch.Tensor


def load_vocabulary(path: Path) -> Vocabulary:
    frame = pd.read_csv(path, sep="\t", header=None, names=["text", "id"]).sort_values("id")
    expected = np.arange(len(frame))
    if not np.array_equal(frame["id"].to_numpy(), expected):
        raise ValueError(f"IDs in {path} must be contiguous and start at zero")
    return Vocabulary(
        text_to_id=dict(zip(frame["text"], frame["id"], strict=True)),
        id_to_text=dict(zip(frame["id"], frame["text"], strict=True)),
    )


def load_events(path: Path) -> pd.DataFrame:
    """Load ICEWS events: subject, relation, object, timestamp, optional label."""
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] < 4:
        raise ValueError(f"Expected at least four columns in {path}, found {raw.shape[1]}")
    events = raw.iloc[:, :4].copy()
    events.columns = ["subject", "relation", "object", "time"]
    return events.astype(np.int64)


def prepare_snapshot(
    positive_triples: np.ndarray,
    negative_triples: np.ndarray,
    num_entities: int,
    *,
    test_size: float,
    validation_size: float,
    random_seed: int,
    device: torch.device,
) -> PreparedSnapshot:
    if len(positive_triples) < 2 or len(negative_triples) < 2:
        raise ValueError("A snapshot needs at least two positive and two negative triples")

    positive_train_validation, positive_test = train_test_split(
        np.arange(len(positive_triples)), test_size=test_size, random_state=random_seed
    )
    positive_train, positive_validation = train_test_split(
        positive_train_validation, test_size=validation_size, random_state=random_seed
    )
    negative_offset = len(positive_triples)
    negative_train_validation, negative_test = train_test_split(
        np.arange(len(negative_triples)) + negative_offset,
        test_size=test_size,
        random_state=random_seed,
    )
    negative_train, negative_validation = train_test_split(
        negative_train_validation, test_size=validation_size, random_state=random_seed
    )

    graph_triples = positive_triples[positive_train]
    graph = Data(
        edge_index=torch.as_tensor(graph_triples[:, [0, 2]].T, dtype=torch.long),
        edge_attr=torch.as_tensor(graph_triples[:, 1], dtype=torch.long),
        num_nodes=num_entities,
    )
    triples_numpy = np.concatenate([positive_triples, negative_triples]).astype(np.int64)
    labels = np.concatenate(
        [np.ones(len(positive_triples)), np.zeros(len(negative_triples))]
    ).astype(np.float32)
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    validation_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[np.concatenate([positive_train, negative_train])] = True
    validation_mask[np.concatenate([positive_validation, negative_validation])] = True
    test_mask[np.concatenate([positive_test, negative_test])] = True

    return PreparedSnapshot(
        graph=graph.to(device),
        triples=torch.as_tensor(triples_numpy, dtype=torch.long, device=device),
        triples_numpy=triples_numpy,
        labels=torch.as_tensor(labels, device=device),
        train_mask=train_mask.to(device),
        validation_mask=validation_mask.to(device),
        test_mask=test_mask.to(device),
    )
