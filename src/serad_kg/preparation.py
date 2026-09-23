from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .anomalies import corrupt_triples
from .anomaly_cache import anomaly_cache_path, load_anomaly_cache
from .data import PreparedSnapshot, load_events, load_vocabulary

Protocol = Literal["pooled", "chronological"]
NegativeSampling = Literal["random", "cache"]
MANIFEST_COLUMNS = ("subject", "relation", "object", "label", "split", "source_time")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparationConfig:
    data_dir: Path
    output_dir: Path
    protocol: Protocol
    negative_sampling: NegativeSampling = "random"
    anomaly_cache_dir: Path | None = None
    max_snapshots: int = 10
    anomalies_per_snapshot: int = 35
    seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.25
    chronological_train_snapshots: int = 7
    chronological_validation_snapshots: int = 1


def _deduplicate_with_first_time(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values("time", kind="stable")
        .drop_duplicates(["subject", "relation", "object"], keep="first")
        .reset_index(drop=True)
    )


def _random_split(size: int, config: PreparationConfig) -> np.ndarray:
    if size < 3:
        raise ValueError("At least three examples are required to create all three splits")
    indices = np.arange(size)
    train_validation, test = train_test_split(
        indices, test_size=config.test_size, random_state=config.seed
    )
    train, validation = train_test_split(
        train_validation, test_size=config.validation_size, random_state=config.seed
    )
    result = np.empty(size, dtype=object)
    result[train] = "train"
    result[validation] = "validation"
    result[test] = "test"
    return result


def _chronological_split(snapshot_index: pd.Series, config: PreparationConfig) -> np.ndarray:
    validation_start = config.chronological_train_snapshots
    test_start = validation_start + config.chronological_validation_snapshots
    if test_start >= config.max_snapshots:
        raise ValueError("The chronological protocol must reserve at least one test snapshot")
    return np.select(
        [snapshot_index < validation_start, snapshot_index < test_start],
        ["train", "validation"],
        default="test",
    )


def _load_negatives(
    config: PreparationConfig,
    snapshots: list[pd.DataFrame],
    timestamps: list[int],
    all_positives: set[tuple[int, int, int]],
    num_entities: int,
    num_relations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, int]] = []
    for index, (snapshot, timestamp) in enumerate(zip(snapshots, timestamps, strict=True)):
        positives = snapshot[["subject", "relation", "object"]].to_numpy(dtype=np.int64)
        if config.negative_sampling == "cache":
            if config.anomaly_cache_dir is None:
                raise ValueError("anomaly_cache_dir is required for cache negative sampling")
            negatives = load_anomaly_cache(
                anomaly_cache_path(config.anomaly_cache_dir, index),
                snapshot_index=index,
                num_entities=num_entities,
                num_relations=num_relations,
            )
        else:
            # Oversampling compensates for candidates rejected against positives from
            # another snapshot or against an earlier generated negative.
            negatives = corrupt_triples(
                positives,
                max(config.anomalies_per_snapshot * 2, 3),
                num_entities,
                num_relations,
                rng,
            )
        accepted = 0
        for subject, relation, object_ in negatives:
            triple = (int(subject), int(relation), int(object_))
            if triple in all_positives:
                continue
            rows.append(
                {
                    "subject": triple[0],
                    "relation": triple[1],
                    "object": triple[2],
                    "source_time": timestamp,
                    "snapshot_index": index,
                }
            )
            accepted += 1
            if config.negative_sampling == "random" and accepted == config.anomalies_per_snapshot:
                break
    negatives = pd.DataFrame(rows)
    negatives = negatives.drop_duplicates(["subject", "relation", "object"], keep="first")
    if len(negatives) < 3:
        raise ValueError("Fewer than three valid, distinct anomalies were prepared")
    return negatives.reset_index(drop=True)


def prepare_experiment(config: PreparationConfig) -> pd.DataFrame:
    """Create the immutable, leakage-free manifest consumed by every model."""
    if config.max_snapshots < 3:
        raise ValueError("max_snapshots must be at least 3")
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    events = load_events(config.data_dir / "train.txt")
    timestamps = [int(value) for value in sorted(events["time"].unique())[: config.max_snapshots]]
    snapshots = [events.loc[events["time"].eq(timestamp)].copy() for timestamp in timestamps]
    selected = pd.concat(snapshots, ignore_index=True)
    positives = _deduplicate_with_first_time(selected)
    time_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    positives["snapshot_index"] = positives["time"].map(time_to_index).astype(np.int64)
    positives = positives.rename(columns={"time": "source_time"})

    positive_set = set(
        map(tuple, positives[["subject", "relation", "object"]].to_numpy(dtype=np.int64))
    )
    negatives = _load_negatives(
        config,
        snapshots,
        timestamps,
        positive_set,
        len(entities.id_to_text),
        len(relations.id_to_text),
    )

    if config.protocol == "pooled":
        positives["split"] = _random_split(len(positives), config)
        negatives["split"] = _random_split(len(negatives), config)
    elif config.protocol == "chronological":
        positives["split"] = _chronological_split(positives["snapshot_index"], config)
        negatives["split"] = _chronological_split(negatives["snapshot_index"], config)
    else:
        raise ValueError(f"Unknown protocol: {config.protocol}")

    positives["label"] = 1
    negatives["label"] = 0
    manifest = pd.concat([positives, negatives], ignore_index=True)[list(MANIFEST_COLUMNS)]
    manifest = manifest.astype(
        {
            "subject": "int64",
            "relation": "int64",
            "object": "int64",
            "label": "int64",
            "source_time": "int64",
        }
    )
    for split in SPLITS:
        split_frame = manifest.loc[manifest["split"].eq(split)]
        if split_frame["label"].nunique() != 2:
            raise ValueError(f"The {split} split does not contain both classes")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "format_version": 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "timestamps": timestamps,
        "counts": {
            split: {
                "positive": int(((manifest["split"] == split) & (manifest["label"] == 1)).sum()),
                "anomaly": int(((manifest["split"] == split) & (manifest["label"] == 0)).sum()),
            }
            for split in SPLITS
        },
        "manifest_sha256": digest,
    }
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return manifest


def load_prepared_experiment(
    directory: Path, *, num_entities: int, num_relations: int, device: torch.device
) -> tuple[PreparedSnapshot, dict[str, object]]:
    manifest_path = directory / "manifest.csv"
    metadata_path = directory / "metadata.json"
    manifest = pd.read_csv(manifest_path)
    if list(manifest.columns) != list(MANIFEST_COLUMNS):
        raise ValueError(f"Unexpected manifest columns in {manifest_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != metadata.get("manifest_sha256"):
        raise ValueError(f"Manifest checksum does not match {metadata_path}")
    if set(manifest["split"]) != set(SPLITS):
        raise ValueError("Prepared manifest must contain train, validation, and test")
    triples = manifest[["subject", "relation", "object"]].to_numpy(dtype=np.int64)
    if np.any(triples[:, [0, 2]] < 0) or np.any(triples[:, [0, 2]] >= num_entities):
        raise ValueError("Entity identifier outside the vocabulary")
    if np.any(triples[:, 1] < 0) or np.any(triples[:, 1] >= num_relations):
        raise ValueError("Relation identifier outside the vocabulary")
    labels_numpy = manifest["label"].to_numpy(dtype=np.float32)
    masks = {
        split: torch.as_tensor(manifest["split"].eq(split).to_numpy(copy=True), device=device)
        for split in SPLITS
    }
    graph_mask = manifest["split"].eq("train") & manifest["label"].eq(1)
    graph_triples = triples[graph_mask.to_numpy()]
    if not len(graph_triples):
        raise ValueError("The prepared experiment has no positive training triple")
    graph = Data(
        edge_index=torch.as_tensor(graph_triples[:, [0, 2]].T, dtype=torch.long),
        edge_attr=torch.as_tensor(graph_triples[:, 1], dtype=torch.long),
        num_nodes=num_entities,
    ).to(device)
    prepared = PreparedSnapshot(
        graph=graph,
        triples=torch.as_tensor(triples, dtype=torch.long, device=device),
        triples_numpy=triples,
        labels=torch.as_tensor(labels_numpy, device=device),
        train_mask=masks["train"],
        validation_mask=masks["validation"],
        test_mask=masks["test"],
    )
    return prepared, metadata
