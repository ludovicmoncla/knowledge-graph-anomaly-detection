from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from sentence_transformers import SentenceTransformer

from .anomalies import corrupt_triples, generate_anomalies_genai
from .anomaly_cache import anomaly_cache_path, load_anomaly_cache, reject_positive_anomalies
from .config import TrainingConfig
from .data import PreparedSnapshot, Vocabulary, load_events, load_vocabulary, prepare_snapshot
from .evaluation import (
    anomaly_auprc,
    anomaly_classification_metrics,
    anomaly_precision_recall_threshold,
    roc_metrics,
    save_loss_plot,
    save_precision_recall_plot,
    save_roc_plot,
)
from .model import SeradKG
from .preparation import load_prepared_experiment

REPEATED_METRICS = (
    "auc_validation",
    "auprc_validation",
    "auc_test",
    "auprc_test",
    "precision_validation",
    "recall_validation",
    "f1_validation",
    "precision_test",
    "recall_test",
    "f1_test",
    "threshold_validation",
    "epochs",
)


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
    entity_labels = list(
        dict.fromkeys(
            label
            for subject, _, object_ in anomalies
            for label in (subject, object_)
            if label not in entities.text_to_id
        )
    )
    relation_labels = list(
        dict.fromkeys(
            relation for _, relation, _ in anomalies if relation not in relations.text_to_id
        )
    )
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
) -> tuple[SeradKG, list[float], list[float]]:
    model = SeradKG(
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
    model: SeradKG,
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
        _, scores = model(prepared.graph, prepared.triples, prepared.labels, prepared.test_mask)
        descriptions = [
            f"{entities.id_to_text[int(s)]} | {relations.id_to_text[int(r)]} | "
            f"{entities.id_to_text[int(o)]}"
            for s, r, o in prepared.triples_numpy
        ]
        frame = model.score_frame(prepared.graph, prepared.triples, prepared.labels, descriptions)

    test_mask = prepared.test_mask.cpu().numpy()
    test_labels = prepared.labels[prepared.test_mask].cpu().numpy()
    test_scores = scores[prepared.test_mask].cpu().numpy()
    _, test_auc = roc_metrics(test_labels, test_scores)
    test_auprc = anomaly_auprc(test_labels, test_scores)
    save_roc_plot(test_labels, test_scores, "Test ROC", output_dir / "roc_test.png")
    save_precision_recall_plot(
        test_labels, test_scores, "Test precision-recall", output_dir / "pr_test.png"
    )
    validation_mask = prepared.validation_mask.cpu().numpy()
    validation_labels = prepared.labels[prepared.validation_mask].cpu().numpy()
    validation_scores = scores[prepared.validation_mask].cpu().numpy()
    _, validation_auc = roc_metrics(validation_labels, validation_scores)
    threshold = anomaly_precision_recall_threshold(validation_labels, validation_scores)
    validation_auprc = anomaly_auprc(validation_labels, validation_scores)
    validation_classification = anomaly_classification_metrics(
        validation_labels, validation_scores, threshold
    )
    test_classification = anomaly_classification_metrics(test_labels, test_scores, threshold)
    save_roc_plot(
        validation_labels,
        validation_scores,
        "Validation ROC",
        output_dir / "roc_validation.png",
    )
    save_precision_recall_plot(
        validation_labels,
        validation_scores,
        "Validation precision-recall",
        output_dir / "pr_validation.png",
    )
    frame["split"] = np.select(
        [test_mask, validation_mask], ["test", "validation"], default="train"
    )
    frame["predicted_anomaly"] = frame["score"] <= threshold
    frame.to_csv(output_dir / "scores.csv", index=False)
    torch.save(model.state_dict(), output_dir / "model.pt")
    return {
        "auc_validation": validation_auc,
        "auprc_validation": validation_auprc,
        "auc_test": test_auc,
        "auprc_test": test_auprc,
        "precision_validation": validation_classification["precision"],
        "recall_validation": validation_classification["recall"],
        "f1_validation": validation_classification["f1"],
        "precision_test": test_classification["precision"],
        "recall_test": test_classification["recall"],
        "f1_test": test_classification["f1"],
        "threshold_validation": threshold,
        "epochs": len(train_losses),
    }


def run(config: TrainingConfig) -> pd.DataFrame:
    if config.prepared_data_dir is not None:
        return run_prepared(config)
    run_started_at = time.perf_counter()
    if config.negative_sampling != "cache" and config.anomaly_count < 3:
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
        "anomaly_cache_dir": (
            str(config.anomaly_cache_dir) if config.anomaly_cache_dir is not None else None
        ),
        "resolved_device": str(device),
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )

    print("SERAD-KG — Knowledge Graph Anomaly Detection", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        f"Entities: {len(entities.id_to_text):,} | "
        f"Relations: {len(relations.id_to_text):,} | "
        f"Training events: {len(events):,}",
        flush=True,
    )
    print(f"Output directory: {config.output_dir}", flush=True)
    openrouter_api_key: str | None = None
    if config.negative_sampling == "cache":
        if config.anomaly_cache_dir is None:
            raise ValueError("--anomaly-cache-dir is required with --negative-sampling cache")
        print(f"Negative sampling: shared cache ({config.anomaly_cache_dir})", flush=True)
    elif config.negative_sampling == "genai":
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
        print(f"  Positives: {len(positives):,}", flush=True)
        if config.negative_sampling == "cache":
            cache_file = anomaly_cache_path(config.anomaly_cache_dir or Path(), snapshot_number - 1)
            if not cache_file.is_file():
                raise FileNotFoundError(f"Missing anomaly cache for snapshot: {cache_file}")
            negatives = load_anomaly_cache(
                cache_file,
                snapshot_index=snapshot_number - 1,
                num_entities=len(entities.id_to_text),
                num_relations=len(relations.id_to_text),
                id_to_entity=entities.id_to_text,
                id_to_relation=relations.id_to_text,
            )
            filtered_negatives = reject_positive_anomalies(negatives, positives)
            if not np.array_equal(filtered_negatives, negatives):
                raise ValueError(
                    f"Anomaly cache contains a positive or duplicate triple: {cache_file}"
                )
            print(f"  Cached negatives loaded: {len(negatives):,}", flush=True)
        elif config.negative_sampling == "genai":
            print(f"  Requested generated negatives: {config.anomaly_count:,}", flush=True)
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
            print(f"  Requested corrupted negatives: {config.anomaly_count:,}", flush=True)
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
        print(
            f"  Test AUC: {metrics['auc_test']:.3f} | Test AUPRC: {metrics['auprc_test']:.3f}",
            flush=True,
        )
        print(
            f"  Test precision: {metrics['precision_test']:.3f} | "
            f"recall: {metrics['recall_test']:.3f} | F1: {metrics['f1_test']:.3f}",
            flush=True,
        )
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


def run_prepared(config: TrainingConfig) -> pd.DataFrame:
    """Train once on the exact split manifest used by every comparison model."""
    run_started_at = time.perf_counter()
    set_random_seed(config.random_seed)
    device = config.resolved_device()
    entities = load_vocabulary(config.data_dir / "entity2id.txt")
    relations = load_vocabulary(config.data_dir / "relation2id.txt")
    prepared, preparation_metadata = load_prepared_experiment(
        config.prepared_data_dir or Path(),
        num_entities=len(entities.id_to_text),
        num_relations=len(relations.id_to_text),
        device=device,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = asdict(config) | {
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "prepared_data_dir": str(config.prepared_data_dir),
        "env_file": str(config.env_file),
        "anomaly_cache_dir": (
            str(config.anomaly_cache_dir) if config.anomaly_cache_dir is not None else None
        ),
        "resolved_device": str(device),
        "prepared_manifest_sha256": preparation_metadata["manifest_sha256"],
    }
    (config.output_dir / "config.json").write_text(
        json.dumps(serializable_config, indent=2), encoding="utf-8"
    )
    print("SERAD-KG — prepared static-graph experiment", flush=True)
    print(f"Prepared data: {config.prepared_data_dir}", flush=True)
    print(f"Device: {device} | examples: {len(prepared.labels):,}", flush=True)
    encoder = SentenceTransformer(config.embedding_model, device=str(device))
    entity_embeddings = encode_vocabulary(encoder, entities, device)
    relation_embeddings = encode_vocabulary(encoder, relations, device)
    model, train_losses, validation_losses = train_model(
        prepared, entity_embeddings, relation_embeddings, config
    )
    metrics = evaluate_snapshot(
        model, prepared, entities, relations, config.output_dir, train_losses, validation_losses
    )
    protocol = str(preparation_metadata["config"]["protocol"])
    summary = pd.DataFrame([{"snapshot": protocol, "status": "ok", **metrics}])
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    print(
        f"Completed {protocol}: test AUROC={metrics['auc_test']:.4f}, "
        f"AUPRC={metrics['auprc_test']:.4f}, precision={metrics['precision_test']:.4f}, "
        f"recall={metrics['recall_test']:.4f}, F1={metrics['f1_test']:.4f} "
        f"in {time.perf_counter() - run_started_at:.1f}s",
        flush=True,
    )
    return summary


def aggregate_repeated_metrics(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    grouped = results.groupby(group_columns, sort=True) if group_columns else [((), results)]
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, float | int] = dict(zip(group_columns, keys, strict=True))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in REPEATED_METRICS:
            values = group[metric].dropna().astype(float)
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            standard_deviation = float(values.std(ddof=1)) if count > 1 else float("nan")
            if count > 1:
                critical_value = float(student_t.ppf(0.975, df=count - 1))
                half_width = critical_value * standard_deviation / np.sqrt(count)
                lower, upper = mean - half_width, mean + half_width
            else:
                lower, upper = float("nan"), float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def run_repeated(config: TrainingConfig, seeds: list[int]) -> dict[str, pd.DataFrame]:
    unique_seeds = list(dict.fromkeys(seeds))
    if len(unique_seeds) < 2:
        raise ValueError("At least two distinct seeds are required for a repeated experiment")
    if any(seed < 0 for seed in unique_seeds):
        raise ValueError("Seeds must be non-negative integers")

    root_output = config.output_dir
    root_output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for run_number, seed in enumerate(unique_seeds, start=1):
        print(
            f"\nRepeated SERAD-KG run {run_number}/{len(unique_seeds)} — seed {seed}",
            flush=True,
        )
        seed_config = replace(
            config,
            output_dir=root_output / f"seed_{seed}",
            random_seed=seed,
        )
        seed_result = run(seed_config).copy()
        seed_result.insert(0, "seed", seed)
        all_results.append(seed_result)

    by_seed = pd.concat(all_results, ignore_index=True)
    by_seed.to_csv(root_output / "summary_by_seed.csv", index=False)
    completed = by_seed[by_seed["status"].eq("ok")].copy()
    if completed.empty:
        raise RuntimeError("No snapshot completed successfully in the repeated experiment")

    by_snapshot = aggregate_repeated_metrics(completed, ["snapshot"])
    by_snapshot.to_csv(root_output / "summary_aggregate.csv", index=False)
    per_seed = completed.groupby("seed", as_index=False)[list(REPEATED_METRICS)].mean()
    overall = aggregate_repeated_metrics(per_seed, [])
    overall.to_csv(root_output / "overall_aggregate.csv", index=False)
    (root_output / "seeds.json").write_text(
        json.dumps({"seeds": unique_seeds}, indent=2), encoding="utf-8"
    )

    print("\nRepeated SERAD-KG experiment completed", flush=True)
    print(f"  Per-seed results: {root_output / 'summary_by_seed.csv'}", flush=True)
    print(f"  Per-snapshot aggregate: {root_output / 'summary_aggregate.csv'}", flush=True)
    print(f"  Overall aggregate: {root_output / 'overall_aggregate.csv'}", flush=True)
    return {"by_seed": by_seed, "by_snapshot": by_snapshot, "overall": overall}


def _alpha_directory_name(alpha: float) -> str:
    return f"alpha_{alpha:.12g}"


def run_alpha_sweep(
    config: TrainingConfig,
    alphas: list[float],
    seeds: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run SERAD-KG for several local/global score weights."""
    unique_alphas = list(dict.fromkeys(alphas))
    if len(unique_alphas) < 2:
        raise ValueError("At least two distinct alpha values are required for an alpha sweep")
    if any(not 0.0 <= alpha <= 1.0 for alpha in unique_alphas):
        raise ValueError("Alpha values must be between 0 and 1")

    root_output = config.output_dir
    root_output.mkdir(parents=True, exist_ok=True)
    all_results = []

    for run_number, alpha in enumerate(unique_alphas, start=1):
        print(
            f"\nSERAD-KG alpha run {run_number}/{len(unique_alphas)} — alpha {alpha:g}",
            flush=True,
        )
        alpha_config = replace(
            config,
            output_dir=root_output / _alpha_directory_name(alpha),
            plausibility_weight=alpha,
        )
        if seeds is None:
            alpha_result = run(alpha_config).copy()
            alpha_result.insert(0, "seed", config.random_seed)
        else:
            alpha_result = run_repeated(alpha_config, seeds)["by_seed"].copy()
        alpha_result.insert(0, "alpha", alpha)
        all_results.append(alpha_result)

    by_alpha = pd.concat(all_results, ignore_index=True)
    summary_path = root_output / (
        "summary_by_alpha.csv" if seeds is None else "summary_by_alpha_and_seed.csv"
    )
    by_alpha.to_csv(summary_path, index=False)
    (root_output / "alphas.json").write_text(
        json.dumps({"alphas": unique_alphas}, indent=2), encoding="utf-8"
    )

    results = {"by_alpha": by_alpha}
    if seeds is not None:
        completed = by_alpha[by_alpha["status"].eq("ok")].copy()
        if completed.empty:
            raise RuntimeError("No snapshot completed successfully in the alpha sweep")
        per_seed = completed.groupby(["alpha", "seed"], as_index=False)[
            list(REPEATED_METRICS)
        ].mean()
        aggregate = aggregate_repeated_metrics(per_seed, ["alpha"])
        aggregate.to_csv(root_output / "alpha_aggregate.csv", index=False)
        results["aggregate"] = aggregate

    print("\nSERAD-KG alpha sweep completed", flush=True)
    print(f"  Results: {summary_path}", flush=True)
    return results
