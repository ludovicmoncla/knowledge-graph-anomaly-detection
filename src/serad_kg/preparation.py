from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from .anomalies import corrupt_triples, generate_anomalies_genai
from .anomaly_cache import anomaly_cache_path, load_anomaly_cache, save_anomaly_cache
from .data import PreparedSnapshot, Vocabulary, load_events, load_vocabulary

Protocol = Literal["pooled", "chronological"]
NegativeSampling = Literal["random", "cache", "genai"]
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
    max_anomaly_ratio: float | None = None
    evaluation_anomaly_ratio: float | None = None
    train_anomaly_ratios: tuple[float, ...] = ()
    openrouter_model: str = "openai/gpt-4.1"
    env_file: Path = Path(".env")
    genai_batch_size: int = 50
    resume_genai: bool = False


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


def _anomaly_count(positive_count: int, ratio: float) -> int:
    """Return the largest anomaly count that does not exceed the requested contamination."""
    return math.floor(positive_count * ratio / (1.0 - ratio))


def _validate_ratio_config(config: PreparationConfig) -> None:
    if config.negative_sampling not in ("random", "cache", "genai"):
        raise ValueError(f"Unknown negative sampling strategy: {config.negative_sampling}")
    if config.genai_batch_size < 1:
        raise ValueError("genai_batch_size must be at least 1")
    if config.max_anomaly_ratio is None:
        if config.train_anomaly_ratios or config.evaluation_anomaly_ratio is not None:
            raise ValueError(
                "train_anomaly_ratios and evaluation_anomaly_ratio require "
                "max_anomaly_ratio"
            )
        return
    if not 0.0 < config.max_anomaly_ratio <= 0.20:
        raise ValueError("max_anomaly_ratio must be greater than 0 and at most 0.20")
    if (
        config.evaluation_anomaly_ratio is not None
        and not 0.0 < config.evaluation_anomaly_ratio <= 0.20
    ):
        raise ValueError("evaluation_anomaly_ratio must be greater than 0 and at most 0.20")
    if not config.train_anomaly_ratios:
        return
    if len(set(config.train_anomaly_ratios)) != len(config.train_anomaly_ratios):
        raise ValueError("train_anomaly_ratios must not contain duplicates")
    for ratio in config.train_anomaly_ratios:
        if not 0.0 < ratio <= config.max_anomaly_ratio:
            raise ValueError(
                "Every train anomaly ratio must be greater than 0 and no larger than "
                "max_anomaly_ratio"
            )


def _allocate_counts(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate an exact total proportionally, using the largest-remainder method."""
    if total == 0:
        return np.zeros(len(weights), dtype=np.int64)
    if len(weights) == 0 or weights.sum() == 0:
        raise ValueError("Cannot allocate anomalies without positive snapshots")
    exact = weights.astype(float) / weights.sum() * total
    allocated = np.floor(exact).astype(np.int64)
    remainder = total - int(allocated.sum())
    if remainder:
        order = np.argsort(-(exact - allocated), kind="stable")
        allocated[order[:remainder]] += 1
    return allocated


def _load_negatives(
    config: PreparationConfig,
    snapshots: list[pd.DataFrame],
    timestamps: list[int],
    all_positives: set[tuple[int, int, int]],
    entities: Vocabulary,
    relations: Vocabulary,
    requested_counts: np.ndarray | None = None,
    openrouter_api_key: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for index, (snapshot, timestamp) in enumerate(zip(snapshots, timestamps, strict=True)):
        positives = snapshot[["subject", "relation", "object"]].to_numpy(dtype=np.int64)
        snapshot_row_start = len(rows)
        cache_directory = config.anomaly_cache_dir or (
            config.output_dir / "generated_anomalies"
        )
        if config.negative_sampling == "cache":
            if requested_counts is not None and int(requested_counts[index]) == 0:
                continue
            if config.anomaly_cache_dir is None:
                raise ValueError("anomaly_cache_dir is required for cache negative sampling")
            negatives = load_anomaly_cache(
                anomaly_cache_path(config.anomaly_cache_dir, index),
                snapshot_index=index,
                num_entities=len(entities.id_to_text),
                num_relations=len(relations.id_to_text),
            )
        else:
            target = (
                int(requested_counts[index])
                if requested_counts is not None
                else config.anomalies_per_snapshot
            )
            if target == 0:
                continue
            resume_path = anomaly_cache_path(cache_directory, index)
            if config.negative_sampling == "genai" and config.resume_genai and resume_path.is_file():
                negatives = load_anomaly_cache(
                    resume_path,
                    snapshot_index=index,
                    num_entities=len(entities.id_to_text),
                    num_relations=len(relations.id_to_text),
                    id_to_entity=entities.id_to_text,
                    id_to_relation=relations.id_to_text,
                )
                print(
                    f"Resuming snapshot {index + 1}/{len(snapshots)} from "
                    f"{len(negatives)} cached anomalies",
                    flush=True,
                )
            else:
                negatives = np.empty((0, 3), dtype=np.int64)
        accepted = 0
        target = None if config.negative_sampling == "cache" else target
        attempts = 0
        process_existing = len(negatives) > 0
        while target is None or accepted < target:
            if process_existing:
                process_existing = False
            elif config.negative_sampling == "random":
                remaining = target - accepted
                negatives = corrupt_triples(
                    positives,
                    max(remaining * 2, 2),
                    len(entities.id_to_text),
                    len(relations.id_to_text),
                    rng,
                )
                attempts += 1
            elif config.negative_sampling == "genai":
                remaining = target - accepted
                batch_size = min(remaining, config.genai_batch_size)
                context_triples = [
                    (
                        entities.id_to_text[int(subject)],
                        relations.id_to_text[int(relation)],
                        entities.id_to_text[int(object_)],
                    )
                    for subject, relation, object_ in positives
                ]
                call_seed = config.seed + (index * 100_000) + attempts
                print(
                    f"Generating snapshot {index + 1}/{len(snapshots)} with "
                    f"{config.openrouter_model}: {accepted}/{target} accepted, "
                    f"requesting {batch_size}",
                    flush=True,
                )
                generated = generate_anomalies_genai(
                    batch_size,
                    context_triples,
                    api_key=openrouter_api_key or "",
                    model=config.openrouter_model,
                    seed=call_seed,
                    require_context_vocabulary=True,
                )
                negatives = np.asarray(
                    [
                        (
                            entities.text_to_id[subject],
                            relations.text_to_id[relation],
                            entities.text_to_id[object_],
                        )
                        for subject, relation, object_ in generated
                    ],
                    dtype=np.int64,
                )
                attempts += 1
            for subject, relation, object_ in negatives:
                triple = (int(subject), int(relation), int(object_))
                if triple in all_positives or triple in seen:
                    continue
                seen.add(triple)
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
                if target is not None and accepted == target:
                    break
            if config.negative_sampling == "genai" and accepted:
                generated_rows = rows[snapshot_row_start:]
                generated_ids = np.asarray(
                    [
                        (row["subject"], row["relation"], row["object"])
                        for row in generated_rows
                    ],
                    dtype=np.int64,
                )
                save_anomaly_cache(
                    anomaly_cache_path(cache_directory, index),
                    generated_ids,
                    snapshot_index=index,
                    source_time=timestamp,
                    id_to_entity=entities.id_to_text,
                    id_to_relation=relations.id_to_text,
                    generator="openrouter",
                    model=config.openrouter_model,
                )
            if target is None:
                break
            if attempts >= 100 and accepted < target:
                raise RuntimeError(
                    f"Could not generate {target} distinct anomalies for snapshot {index}"
                )
    negatives = pd.DataFrame(rows)
    if len(negatives) < 3:
        raise ValueError("Fewer than three valid, distinct anomalies were prepared")
    return negatives.reset_index(drop=True)


def _ratio_negative_counts(
    positives: pd.DataFrame,
    ratio: float,
    evaluation_ratio: float | None = None,
) -> dict[str, int]:
    counts = positives["split"].value_counts()
    ratios = {split: ratio for split in SPLITS}
    if evaluation_ratio is not None:
        ratios["validation"] = evaluation_ratio
        ratios["test"] = evaluation_ratio
    result = {
        split: _anomaly_count(int(counts.get(split, 0)), ratios[split])
        for split in SPLITS
    }
    if any(count < 1 for count in result.values()):
        raise ValueError(
            "The selected snapshots contain too few positives to put an anomaly in every split "
            "at the requested ratios"
        )
    return result


def _requested_counts_by_snapshot(
    positives: pd.DataFrame,
    target_by_split: dict[str, int],
    snapshot_count: int,
    protocol: Protocol,
) -> np.ndarray:
    result = np.zeros(snapshot_count, dtype=np.int64)
    if protocol == "pooled":
        weights = (
            positives["snapshot_index"]
            .value_counts()
            .reindex(range(snapshot_count), fill_value=0)
            .to_numpy()
        )
        return _allocate_counts(weights, sum(target_by_split.values()))
    for split in SPLITS:
        split_positives = positives.loc[positives["split"].eq(split)]
        indices = np.sort(split_positives["snapshot_index"].unique())
        weights = (
            split_positives["snapshot_index"]
            .value_counts()
            .reindex(indices, fill_value=0)
            .to_numpy()
        )
        result[indices] = _allocate_counts(weights, target_by_split[split])
    return result


def _select_ratio_negatives(
    negatives: pd.DataFrame,
    target_by_split: dict[str, int],
    config: PreparationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    candidates = negatives.copy()
    if config.protocol == "pooled":
        order = rng.permutation(len(candidates))
        candidates = candidates.iloc[order].reset_index(drop=True)
        boundaries = np.cumsum([target_by_split[split] for split in SPLITS])
        if len(candidates) < int(boundaries[-1]):
            raise ValueError(
                f"Only {len(candidates)} distinct anomalies are available; "
                f"{int(boundaries[-1])} are required for max_anomaly_ratio"
            )
        candidates = candidates.iloc[: int(boundaries[-1])].copy()
        candidates["split"] = np.repeat(
            SPLITS, [target_by_split[split] for split in SPLITS]
        )
        return candidates

    candidates["split"] = _chronological_split(candidates["snapshot_index"], config)
    selected = []
    for split in SPLITS:
        split_candidates = candidates.loc[candidates["split"].eq(split)]
        target = target_by_split[split]
        if len(split_candidates) < target:
            raise ValueError(
                f"Only {len(split_candidates)} distinct {split} anomalies are available; "
                f"{target} are required for max_anomaly_ratio"
            )
        order = rng.permutation(len(split_candidates))[:target]
        selected.append(split_candidates.iloc[order])
    return pd.concat(selected, ignore_index=True)


def _ratio_directory_name(ratio: float) -> str:
    return f"train_ratio_{ratio:g}"


def _write_prepared_experiment(
    directory: Path,
    manifest: pd.DataFrame,
    config: PreparationConfig,
    timestamps: list[int],
    *,
    train_anomaly_ratio: float | None,
    fixed_evaluation_sha256: str | None = None,
) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata = {
        "format_version": 2 if config.max_anomaly_ratio is not None else 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "timestamps": timestamps,
        "snapshot_count": len(timestamps),
        "train_anomaly_ratio": train_anomaly_ratio,
        "validation_anomaly_ratio": (
            config.evaluation_anomaly_ratio
            if config.evaluation_anomaly_ratio is not None
            else config.max_anomaly_ratio
        ),
        "test_anomaly_ratio": (
            config.evaluation_anomaly_ratio
            if config.evaluation_anomaly_ratio is not None
            else config.max_anomaly_ratio
        ),
        "counts": {
            split: {
                "positive": int(((manifest["split"] == split) & (manifest["label"] == 1)).sum()),
                "anomaly": int(((manifest["split"] == split) & (manifest["label"] == 0)).sum()),
            }
            for split in SPLITS
        },
        "manifest_sha256": digest,
    }
    if fixed_evaluation_sha256 is not None:
        metadata["fixed_evaluation_sha256"] = fixed_evaluation_sha256
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return digest


def prepare_experiment(config: PreparationConfig) -> pd.DataFrame:
    """Create the immutable, leakage-free manifest consumed by every model."""
    if config.max_snapshots < 3:
        raise ValueError("max_snapshots must be at least 3")
    _validate_ratio_config(config)
    openrouter_api_key: str | None = None
    if config.negative_sampling == "genai":
        from dotenv import load_dotenv

        load_dotenv(config.env_file, override=False)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY was not found in {config.env_file} or the environment"
            )
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    events = load_events(config.data_dir / "train.txt")
    timestamps = [int(value) for value in sorted(events["time"].unique())[: config.max_snapshots]]
    if len(timestamps) != config.max_snapshots:
        raise ValueError(
            f"Requested {config.max_snapshots} snapshots, but {len(timestamps)} are available"
        )
    snapshots = [events.loc[events["time"].eq(timestamp)].copy() for timestamp in timestamps]
    selected = pd.concat(snapshots, ignore_index=True)
    positives = _deduplicate_with_first_time(selected)
    time_to_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    positives["snapshot_index"] = positives["time"].map(time_to_index).astype(np.int64)
    positives = positives.rename(columns={"time": "source_time"})

    positive_set = set(
        map(tuple, positives[["subject", "relation", "object"]].to_numpy(dtype=np.int64))
    )
    target_by_split: dict[str, int] | None = None
    requested_counts: np.ndarray | None = None
    if config.protocol == "pooled":
        positives["split"] = _random_split(len(positives), config)
    elif config.protocol == "chronological":
        positives["split"] = _chronological_split(positives["snapshot_index"], config)
    else:
        raise ValueError(f"Unknown protocol: {config.protocol}")
    if config.max_anomaly_ratio is not None:
        target_by_split = _ratio_negative_counts(
            positives,
            config.max_anomaly_ratio,
            config.evaluation_anomaly_ratio,
        )
        requested_counts = _requested_counts_by_snapshot(
            positives, target_by_split, len(snapshots), config.protocol
        )
    negatives = _load_negatives(
        config,
        snapshots,
        timestamps,
        positive_set,
        entities,
        relations,
        requested_counts,
        openrouter_api_key,
    )

    if target_by_split is not None:
        negatives = _select_ratio_negatives(negatives, target_by_split, config)
    elif config.protocol == "pooled":
        negatives["split"] = _random_split(len(negatives), config)
    else:
        negatives["split"] = _chronological_split(negatives["snapshot_index"], config)

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

    _write_prepared_experiment(
        config.output_dir,
        manifest,
        config,
        timestamps,
        train_anomaly_ratio=config.max_anomaly_ratio,
    )
    if config.train_anomaly_ratios:
        evaluation = manifest.loc[manifest["split"].isin(("validation", "test"))]
        evaluation_sha256 = hashlib.sha256(
            evaluation.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        train_positives = manifest.loc[
            manifest["split"].eq("train") & manifest["label"].eq(1)
        ]
        train_negatives = manifest.loc[
            manifest["split"].eq("train") & manifest["label"].eq(0)
        ]
        for ratio in sorted(config.train_anomaly_ratios):
            anomaly_count = _anomaly_count(len(train_positives), ratio)
            if anomaly_count < 1:
                raise ValueError(
                    f"Training split is too small to contain an anomaly at ratio {ratio:g}"
                )
            variant = pd.concat(
                [train_positives, train_negatives.iloc[:anomaly_count], evaluation],
                ignore_index=True,
            )
            _write_prepared_experiment(
                config.output_dir / _ratio_directory_name(ratio),
                variant,
                config,
                timestamps,
                train_anomaly_ratio=ratio,
                fixed_evaluation_sha256=evaluation_sha256,
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
        edge_example_index=torch.as_tensor(
            np.flatnonzero(graph_mask.to_numpy()), dtype=torch.long
        ),
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
