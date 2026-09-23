from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_COLUMNS = (
    "dataset",
    "snapshot_index",
    "source_time",
    "subject",
    "relation",
    "object",
    "sub_id",
    "rel_id",
    "obj_id",
    "generator",
    "model",
)


def anomaly_cache_path(directory: Path, snapshot_index: int) -> Path:
    return directory / f"snapshot_{snapshot_index:03d}.csv"


def save_anomaly_cache(
    path: Path,
    anomalies: np.ndarray,
    *,
    snapshot_index: int,
    source_time: int,
    id_to_entity: dict[int, str],
    id_to_relation: dict[int, str],
    generator: str,
    model: str,
) -> None:
    anomaly_ids = np.asarray(anomalies, dtype=np.int64)
    if anomaly_ids.ndim != 2 or anomaly_ids.shape[1] != 3 or len(anomaly_ids) == 0:
        raise ValueError("Anomalies must be a non-empty array with shape (n, 3)")
    if len({tuple(row) for row in anomaly_ids.tolist()}) != len(anomaly_ids):
        raise ValueError("Anomaly cache cannot contain duplicate triples")

    rows = [
        {
            "dataset": "ICEWS18",
            "snapshot_index": snapshot_index,
            "source_time": source_time,
            "subject": id_to_entity[int(subject)],
            "relation": id_to_relation[int(relation)],
            "object": id_to_entity[int(object_)],
            "sub_id": int(subject),
            "rel_id": int(relation),
            "obj_id": int(object_),
            "generator": generator,
            "model": model,
        }
        for subject, relation, object_ in anomaly_ids
    ]
    frame = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def load_anomaly_cache(
    path: Path,
    *,
    snapshot_index: int,
    num_entities: int,
    num_relations: int,
    id_to_entity: dict[int, str] | None = None,
    id_to_relation: dict[int, str] | None = None,
) -> np.ndarray:
    frame = pd.read_csv(path)
    missing = set(CACHE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Invalid anomaly cache {path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Invalid anomaly cache {path}: no anomaly")
    if not frame["snapshot_index"].eq(snapshot_index).all():
        raise ValueError(f"Invalid anomaly cache {path}: expected snapshot index {snapshot_index}")

    id_columns = ["sub_id", "rel_id", "obj_id"]
    try:
        anomaly_ids = frame[id_columns].to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid identifiers in anomaly cache {path}") from error

    if not np.array_equal(frame[id_columns].to_numpy(), anomaly_ids):
        raise ValueError(f"Non-integer identifiers in anomaly cache {path}")
    if np.any(anomaly_ids[:, [0, 2]] < 0) or np.any(anomaly_ids[:, [0, 2]] >= num_entities):
        raise ValueError(f"Entity identifier outside the vocabulary in anomaly cache {path}")
    if np.any(anomaly_ids[:, 1] < 0) or np.any(anomaly_ids[:, 1] >= num_relations):
        raise ValueError(f"Relation identifier outside the vocabulary in anomaly cache {path}")
    if len({tuple(row) for row in anomaly_ids.tolist()}) != len(anomaly_ids):
        raise ValueError(f"Duplicate triples in anomaly cache {path}")

    if id_to_entity is not None:
        expected_subjects = [id_to_entity[int(identifier)] for identifier in anomaly_ids[:, 0]]
        expected_objects = [id_to_entity[int(identifier)] for identifier in anomaly_ids[:, 2]]
        if (
            frame["subject"].tolist() != expected_subjects
            or frame["object"].tolist() != expected_objects
        ):
            raise ValueError(f"Entity labels do not match identifiers in anomaly cache {path}")
    if id_to_relation is not None:
        expected_relations = [id_to_relation[int(identifier)] for identifier in anomaly_ids[:, 1]]
        if frame["relation"].tolist() != expected_relations:
            raise ValueError(f"Relation labels do not match identifiers in anomaly cache {path}")

    return anomaly_ids


def reject_positive_anomalies(anomalies: np.ndarray, positives: np.ndarray) -> np.ndarray:
    positive_set = {tuple(row) for row in np.asarray(positives, dtype=np.int64).tolist()}
    unique_negatives = list(
        dict.fromkeys(
            tuple(row)
            for row in np.asarray(anomalies, dtype=np.int64).tolist()
            if tuple(row) not in positive_set
        )
    )
    if len(unique_negatives) < 3:
        raise ValueError(
            "At least three distinct generated anomalies outside the positive snapshot are required"
        )
    return np.asarray(unique_negatives, dtype=np.int64)
