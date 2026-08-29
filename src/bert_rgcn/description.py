from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import Vocabulary, load_events, load_vocabulary


SPLITS = ("train", "valid", "test")


def validate_events(
    events: pd.DataFrame,
    entities: Vocabulary,
    relations: Vocabulary,
    source: Path,
) -> None:
    if events.empty:
        raise ValueError(f"No events found in {source}")
    limits = {
        "subject": len(entities.id_to_text),
        "object": len(entities.id_to_text),
        "relation": len(relations.id_to_text),
    }
    for column, upper_bound in limits.items():
        if events[column].min() < 0 or events[column].max() >= upper_bound:
            raise ValueError(
                f"{source}: {column} IDs must be between 0 and {upper_bound - 1}"
            )


def readable_events(
    events: pd.DataFrame, entities: Vocabulary, relations: Vocabulary
) -> pd.DataFrame:
    result = events.rename(
        columns={"subject": "subject_id", "relation": "relation_id", "object": "object_id"}
    ).copy()
    result.insert(0, "subject", result["subject_id"].map(entities.id_to_text))
    result.insert(1, "relation", result["relation_id"].map(relations.id_to_text))
    result.insert(2, "object", result["object_id"].map(entities.id_to_text))
    return result[
        ["subject", "relation", "object", "time", "subject_id", "relation_id", "object_id"]
    ]


def dataset_summary(
    splits: dict[str, pd.DataFrame], entities: Vocabulary, relations: Vocabulary
) -> dict[str, object]:
    all_events = pd.concat(splits.values(), ignore_index=True)
    node_ids = pd.concat([all_events["subject"], all_events["object"]], ignore_index=True)
    degrees = node_ids.value_counts()
    num_entities = len(entities.id_to_text)
    num_events = len(all_events)
    density = num_events / (num_entities * (num_entities - 1)) if num_entities > 1 else 0.0
    return {
        "num_entities": num_entities,
        "num_relations": len(relations.id_to_text),
        "num_events": num_events,
        "events_per_split": {name: len(frame) for name, frame in splits.items()},
        "time_min": int(all_events["time"].min()),
        "time_max": int(all_events["time"].max()),
        "num_snapshots": int(all_events["time"].nunique()),
        "mean_events_per_snapshot": float(all_events.groupby("time").size().mean()),
        "directed_multigraph_density": float(density),
        "mean_total_degree": float(degrees.sum() / num_entities),
    }


def relation_frequencies(
    events: pd.DataFrame, relations: Vocabulary
) -> pd.DataFrame:
    counts = events["relation"].value_counts().rename_axis("relation_id").reset_index(name="count")
    counts["relation"] = counts["relation_id"].map(relations.id_to_text)
    counts["percentage"] = counts["count"] / len(events) * 100
    return counts[["relation_id", "relation", "count", "percentage"]]


def save_degree_distribution(events: pd.DataFrame, output_path: Path) -> None:
    node_ids = pd.concat([events["subject"], events["object"]], ignore_index=True)
    degree_frequency = node_ids.value_counts().value_counts().sort_index()
    degrees = degree_frequency.index.to_numpy()
    counts = degree_frequency.to_numpy()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(degrees, counts, s=12)
    axes[0].set(xlabel="Degree", ylabel="Entity count", title="Linear scale")
    axes[1].scatter(degrees, counts, s=12)
    axes[1].set(xlabel="Degree", ylabel="Entity count", title="Log-log scale", xscale="log",
                yscale="log")
    for axis in axes:
        axis.grid(linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def describe_dataset(data_dir: Path, output_dir: Path) -> dict[str, object]:
    """Validate ICEWS18 and create documented, human-readable derived artifacts."""
    entities = load_vocabulary(data_dir / "entity2id.txt")
    relations = load_vocabulary(data_dir / "relation2id.txt")
    splits: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        source = data_dir / f"{split}.txt"
        events = load_events(source)
        validate_events(events, entities, relations, source)
        splits[split] = events

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, events in splits.items():
        readable_events(events, entities, relations).to_csv(
            output_dir / f"{split}_readable.csv", index=False
        )

    all_events = pd.concat(splits.values(), ignore_index=True)
    relation_frequencies(all_events, relations).to_csv(
        output_dir / "relation_frequencies.csv", index=False
    )
    save_degree_distribution(all_events, output_dir / "degree_distribution.png")
    summary = dataset_summary(splits, entities, relations)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
