from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .anomalies import corrupt_triples, generate_anomalies_genai
from .config import TrainingConfig
from .data import PreparedSnapshot, Vocabulary, load_events, load_vocabulary, prepare_snapshot
from .evaluation import roc_metrics, save_loss_plot, save_roc_plot
from .model import BertRGCN


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_vocabulary(
    encoder: SentenceTransformer, vocabulary: Vocabulary, device: torch.device
) -> torch.Tensor:
    ordered = [vocabulary.id_to_text[index] for index in range(len(vocabulary.id_to_text))]
    return encoder.encode(ordered, convert_to_tensor=True, device=str(device))


def encode_generated_anomalies(
    anomalies: list[tuple[str, str, str]],
    entities: Vocabulary,
    relations: Vocabulary,
    encoder: SentenceTransformer,
    entity_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    entity_labels = list(dict.fromkeys(
        label for subject, _, object_ in anomalies for label in (subject, object_)
        if label not in entities.text_to_id
    ))
    relation_labels = list(dict.fromkeys(
        relation for _, relation, _ in anomalies if relation not in relations.text_to_id
    ))
    if entity_labels:
        encoded = encoder.encode(entity_labels, convert_to_tensor=True, device=str(device))
        entity_embeddings = torch.cat([entity_embeddings, encoded], dim=0)
        for label in entity_labels:
            identifier = len(entities.id_to_text)
            entities.text_to_id[label] = identifier
            entities.id_to_text[identifier] = label
    if relation_labels:
        encoded = encoder.encode(relation_labels, convert_to_tensor=True, device=str(device))
        relation_embeddings = torch.cat([relation_embeddings, encoded], dim=0)
        for label in relation_labels:
            identifier = len(relations.id_to_text)
            relations.text_to_id[label] = identifier
            relations.id_to_text[identifier] = label
    anomaly_ids = np.asarray(
        [
            [
                entities.text_to_id[subject],
                relations.text_to_id[relation],
                entities.text_to_id[object_],
            ]
            for subject, relation, object_ in anomalies
        ],
        dtype=np.int64,
    )
    return anomaly_ids, entity_embeddings, relation_embeddings


def train_model(
    prepared: PreparedSnapshot,
    entity_embeddings: torch.Tensor,
    relation_embeddings: torch.Tensor,
    config: TrainingConfig,
) -> tuple[BertRGCN, list[float], list[float]]:
    model = BertRGCN(
        entity_embeddings,
        relation_embeddings,
        compressed_dim=config.compressed_dim,
        plausibility_weight=config.plausibility_weight,
    ).to(prepared.triples.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    stopped_early = False
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
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"  Epoch {epoch:03d}/{config.epochs:03d} — "
                f"train loss: {train_losses[-1]:.4f} — "
                f"validation loss: {validation_losses[-1]:.4f}",
                flush=True,
            )
        if validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    if stopped_early:
        print(
            f"  Early stopping after {len(train_losses)} epochs "
            f"(best validation loss: {best_loss:.4f})",
            flush=True,
        )
    else:
        print(
            f"  Training completed after {len(train_losses)} epochs "
            f"(best validation loss: {best_loss:.4f})",
            flush=True,
        )
    return model, train_losses, validation_losses


def evaluate_snapshot(
    model: BertRGCN,
    prepared: PreparedSnapshot,
    entities: Vocabulary,
    relations: Vocabulary,
    output_dir: Path,
    train_losses: list[float],
    validation_losses: list[float],
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_loss_plot(train_losses, validation_losses, output_dir / "loss.png")
    model.eval()
    with torch.no_grad():
        _, scores = model(
            prepared.graph, prepared.triples, prepared.labels, prepared.test_mask
        )
        descriptions = [
            f"{entities.id_to_text[int(s)]} | {relations.id_to_text[int(r)]} | "
            f"{entities.id_to_text[int(o)]}"
            for s, r, o in prepared.triples_numpy
        ]
        frame = model.score_frame(
            prepared.graph, prepared.triples, prepared.labels, descriptions
        )

    test_mask = prepared.test_mask.cpu().numpy()
    test_labels = prepared.labels[prepared.test_mask].cpu().numpy()
    test_scores = scores[prepared.test_mask].cpu().numpy()
    threshold, test_auc = roc_metrics(test_labels, test_scores)
    save_roc_plot(test_labels, test_scores, "Test ROC", output_dir / "roc_test.png")
    validation_mask = prepared.validation_mask.cpu().numpy()
    frame["split"] = np.select(
        [test_mask, validation_mask], ["test", "validation"], default="train"
    )
    frame["predicted_anomaly"] = frame["score"] < threshold
    frame.to_csv(output_dir / "scores.csv", index=False)
    torch.save(model.state_dict(), output_dir / "model.pt")
    return {"auc_test": test_auc, "threshold": threshold, "epochs": len(train_losses)}


def run(config: TrainingConfig) -> pd.DataFrame:
    run_started_at = time.perf_counter()
    if config.anomaly_count < 3:
        raise ValueError("anomaly_count must be at least 3 for train/validation/test splitting")
    set_random_seed(config.random_seed)
    device = config.resolved_device()
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    events = load_events(config.data_dir / "train.txt")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = asdict(config) | {
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "env_file": str(config.env_file),
        "resolved_device": str(device),
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    print("Knowledge Graph Anomaly Detection", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        f"Entities: {len(entities.id_to_text):,} | "
        f"Relations: {len(relations.id_to_text):,} | "
        f"Training events: {len(events):,}",
        flush=True,
    )
    print(f"Output directory: {config.output_dir}", flush=True)
    openrouter_api_key: str | None = None
    if config.negative_sampling == "genai":
        from dotenv import load_dotenv

        load_dotenv(config.env_file, override=False)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY was not found in {config.env_file} or the environment"
            )
        print(
            f"Negative sampling: genai via OpenRouter ({config.openrouter_model})",
            flush=True,
        )
    else:
        print("Negative sampling: random corruption", flush=True)
    print(f"Loading and encoding labels with {config.embedding_model}...", flush=True)
    encoder = SentenceTransformer(config.embedding_model, device=str(device))
    entity_embeddings = encode_vocabulary(encoder, entities, device)
    relation_embeddings = encode_vocabulary(encoder, relations, device)
    print("Semantic embeddings ready.", flush=True)
    rng = np.random.default_rng(config.random_seed)
    times = sorted(events["time"].unique())
    if config.max_snapshots is not None:
        times = times[: config.max_snapshots]
    print(f"Snapshots selected: {len(times)}", flush=True)

    summary: list[dict[str, object]] = []
    for snapshot_number, timestamp in enumerate(times, start=1):
        snapshot_started_at = time.perf_counter()
        positives = events.loc[
            events["time"] == timestamp, ["subject", "relation", "object"]
        ].to_numpy()
        if len(positives) < config.min_triples:
            summary.append(
                {"snapshot": int(timestamp), "status": "skipped", "triples": len(positives)}
            )
            print(
                f"\nSnapshot {snapshot_number}/{len(times)} — timestamp {timestamp}: "
                f"skipped ({len(positives)} triples, minimum {config.min_triples})",
                flush=True,
            )
            continue
        snapshot_output = config.output_dir / f"snapshot_{int(timestamp):04d}"
        print(f"\nSnapshot {snapshot_number}/{len(times)} — timestamp {timestamp}", flush=True)
        print(
            f"  Positives: {len(positives):,} | Generated negatives: {config.anomaly_count:,}",
            flush=True,
        )
        if config.negative_sampling == "genai":
            context_triples = [
                (
                    entities.id_to_text[int(subject)],
                    relations.id_to_text[int(relation)],
                    entities.id_to_text[int(object_)],
                )
                for subject, relation, object_ in positives
            ]
            generated = generate_anomalies_genai(
                config.anomaly_count,
                context_triples,
                api_key=openrouter_api_key or "",
                model=config.openrouter_model,
                seed=config.random_seed + snapshot_number,
            )
            negatives, entity_embeddings, relation_embeddings = encode_generated_anomalies(
                generated,
                entities,
                relations,
                encoder,
                entity_embeddings,
                relation_embeddings,
                device,
            )
            positive_set = {tuple(triple) for triple in positives.tolist()}
            if any(tuple(triple) in positive_set for triple in negatives.tolist()):
                raise ValueError("OpenRouter generated a triple that is positive in this snapshot")
            print(
                f"  Vocabulary after generation: {len(entities.id_to_text):,} entities, "
                f"{len(relations.id_to_text):,} relations",
                flush=True,
            )
        else:
            negatives = corrupt_triples(
                positives,
                config.anomaly_count,
                len(entities.id_to_text),
                len(relations.id_to_text),
                rng,
            )
        prepared = prepare_snapshot(
            positives,
            negatives,
            len(entities.id_to_text),
            test_size=config.test_size,
            validation_size=config.validation_size,
            random_seed=config.random_seed,
            device=device,
        )
        print(
            f"  Split sizes: train={int(prepared.train_mask.sum().item()):,}, "
            f"validation={int(prepared.validation_mask.sum().item()):,}, "
            f"test={int(prepared.test_mask.sum().item()):,}",
            flush=True,
        )
        model, train_losses, validation_losses = train_model(
            prepared, entity_embeddings, relation_embeddings, config
        )
        metrics = evaluate_snapshot(
            model,
            prepared,
            entities,
            relations,
            snapshot_output,
            train_losses,
            validation_losses,
        )
        row = {
            "snapshot": int(timestamp),
            "status": "ok",
            "triples": len(positives),
            **metrics,
        }
        summary.append(row)
        elapsed = time.perf_counter() - snapshot_started_at
        print(f"  Test AUC: {metrics['auc_test']:.3f}", flush=True)
        print(f"  Saved to: {snapshot_output} ({elapsed:.1f} s)", flush=True)

    result = pd.DataFrame(summary)
    summary_path = config.output_dir / "summary.csv"
    result.to_csv(summary_path, index=False)
    completed = sum(row["status"] == "ok" for row in summary)
    skipped = len(summary) - completed
    elapsed = time.perf_counter() - run_started_at
    print("\nRun completed", flush=True)
    print(f"  Snapshots completed: {completed} | skipped: {skipped}", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    print(f"  Total time: {elapsed:.1f} s", flush=True)
    return result
