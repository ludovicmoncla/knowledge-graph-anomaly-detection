from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from ..data import PreparedSnapshot, load_vocabulary
from ..pipeline import (
    aggregate_repeated_metrics,
    evaluate_snapshot,
    load_completed_result,
    set_random_seed,
)
from ..preparation import load_prepared_experiment
from .lognet import LoGNet, build_line_graph_edges


@dataclass(frozen=True)
class LogNetConfig:
    data_dir: Path
    prepared_data_dir: Path
    output_dir: Path = Path("outputs/lognet")
    embedding_dim: int = 16
    plausibility_weight: float = 0.3
    dropout: float = 0.3
    learning_rate: float = 1e-2
    weight_decay: float = 1e-3
    epochs: int = 300
    patience: int = 25
    min_delta: float = 1e-5
    seed: int = 42
    device: str = "auto"
    save_model: bool = True

    def __post_init__(self) -> None:
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def train_model(
    prepared: PreparedSnapshot, num_entities: int, num_relations: int, config: LogNetConfig
) -> tuple[LoGNet, list[float], list[float]]:
    graph_mask = prepared.train_mask & prepared.labels.eq(1)
    edges = build_line_graph_edges(prepared.triples_numpy, graph_mask.detach().cpu().numpy()).to(
        prepared.triples.device
    )
    model = LoGNet(
        num_entities=num_entities,
        num_relations=num_relations,
        line_graph_edge_index=edges,
        embedding_dim=config.embedding_dim,
        plausibility_weight=config.plausibility_weight,
        dropout=config.dropout,
    ).to(prepared.triples.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss, _ = model(
            prepared.graph, prepared.triples, prepared.labels, prepared.train_mask
        )
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss, _ = model(
                prepared.graph, prepared.triples, prepared.labels, prepared.validation_mask
            )
        validation_losses.append(float(validation_loss))
        if validation_losses[-1] < best_loss - config.min_delta:
            best_loss = validation_losses[-1]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:03d}/{config.epochs:03d} — train loss: "
                f"{train_losses[-1]:.4f} — validation loss: {validation_losses[-1]:.4f}",
                flush=True,
            )
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, train_losses, validation_losses


def run(config: LogNetConfig) -> pd.DataFrame:
    started_at = time.perf_counter()
    set_random_seed(config.seed)
    device = config.resolved_device()
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    prepared, metadata = load_prepared_experiment(
        config.prepared_data_dir,
        num_entities=len(entities.id_to_text),
        num_relations=len(relations.id_to_text),
        device=device,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    } | {"resolved_device": str(device), "prepared_manifest_sha256": metadata["manifest_sha256"]}
    (config.output_dir / "config.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )
    print("LoGNet — prepared static-graph experiment", flush=True)
    print(f"Prepared data: {config.prepared_data_dir}", flush=True)
    print(f"Device: {device} | examples: {len(prepared.labels):,}", flush=True)
    model, train_losses, validation_losses = train_model(
        prepared, len(entities.id_to_text), len(relations.id_to_text), config
    )
    metrics = evaluate_snapshot(
        model,
        prepared,
        entities,
        relations,
        config.output_dir,
        train_losses,
        validation_losses,
        save_model=config.save_model,
    )
    elapsed = time.perf_counter() - started_at
    metrics["elapsed_seconds"] = elapsed
    protocol = str(metadata["config"]["protocol"])
    result = pd.DataFrame([{"protocol": protocol, "status": "ok", **metrics}])
    result.to_csv(config.output_dir / "summary.csv", index=False)
    print(
        f"Completed {protocol}: test AUROC={metrics['auc_test']:.4f}, "
        f"AUPRC={metrics['auprc_test']:.4f}, precision={metrics['precision_test']:.4f}, "
        f"recall={metrics['recall_test']:.4f}, F1={metrics['f1_test']:.4f} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return result


def run_repeated(
    config: LogNetConfig, seeds: list[int], *, resume: bool = False
) -> dict[str, pd.DataFrame]:
    unique_seeds = list(dict.fromkeys(seeds))
    if len(unique_seeds) < 2:
        raise ValueError("At least two distinct seeds are required")
    root_output = config.output_dir
    root_output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in unique_seeds:
        seed_config = replace(config, seed=seed, output_dir=root_output / f"seed_{seed}")
        seed_result = load_completed_result(seed_config) if resume else None
        if seed_result is None:
            seed_result = run(seed_config).copy()
        else:
            seed_result = seed_result.copy()
            print(f"Resuming: completed LoGNet seed {seed} reused", flush=True)
        seed_result.insert(0, "seed", seed)
        results.append(seed_result)
    by_seed = pd.concat(results, ignore_index=True)
    by_seed.to_csv(root_output / "summary_by_seed.csv", index=False)
    aggregate = aggregate_repeated_metrics(by_seed[by_seed["status"].eq("ok")], ["protocol"])
    aggregate.to_csv(root_output / "summary_aggregate.csv", index=False)
    return {"by_seed": by_seed, "aggregate": aggregate}
