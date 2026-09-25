from __future__ import annotations

import math

import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


def balanced_bce_with_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average normal and anomaly BCE losses with equal class contributions."""
    normal_mask = labels.eq(1)
    anomaly_mask = labels.eq(0)
    if not bool(normal_mask.any()) or not bool(anomaly_mask.any()):
        raise ValueError("Balanced BCE requires both normal and anomaly examples")
    losses = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return 0.5 * (losses[normal_mask].mean() + losses[anomaly_mask].mean())


def pairwise_ranking_loss(
    plausibility_scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    randomize_pairs: bool,
) -> torch.Tensor:
    """Encourage normal triples to outrank anomalies by the requested margin."""
    normal_scores = plausibility_scores[labels.eq(1)]
    anomaly_scores = plausibility_scores[labels.eq(0)]
    if not len(normal_scores) or not len(anomaly_scores):
        raise ValueError("Pairwise ranking loss requires both normal and anomaly examples")
    pair_count = min(len(normal_scores), len(anomaly_scores))
    if randomize_pairs:
        normal_order = torch.randperm(len(normal_scores), device="cpu")[:pair_count].to(
            plausibility_scores.device
        )
        anomaly_order = torch.randperm(len(anomaly_scores), device="cpu")[:pair_count].to(
            plausibility_scores.device
        )
    else:
        normal_order = torch.arange(pair_count, device=plausibility_scores.device)
        anomaly_order = torch.arange(pair_count, device=plausibility_scores.device)
    score_difference = normal_scores[normal_order] - anomaly_scores[anomaly_order]
    return functional.softplus(margin - score_difference).mean()


class SeradKG(nn.Module):
    """Combine semantic triple scoring with R-GCN/TransE graph scoring."""

    def __init__(
        self,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        compressed_dim: int = 64,
        plausibility_weight: float = 0.5,
        auxiliary_loss_weight: float = 0.25,
        ranking_loss_weight: float = 0.0,
        ranking_margin: float = 0.5,
        score_normalization: str = "dynamic",
    ) -> None:
        super().__init__()
        if not 0.0 <= plausibility_weight <= 1.0:
            raise ValueError("plausibility_weight must be between 0 and 1")
        if auxiliary_loss_weight < 0.0:
            raise ValueError("auxiliary_loss_weight must be non-negative")
        if ranking_loss_weight < 0.0:
            raise ValueError("ranking_loss_weight must be non-negative")
        if ranking_margin < 0.0:
            raise ValueError("ranking_margin must be non-negative")
        if score_normalization not in ("stable", "dynamic"):
            raise ValueError("score_normalization must be 'stable' or 'dynamic'")
        if compressed_dim < 1:
            raise ValueError("compressed_dim must be at least 1")

        self.num_entities = entity_embeddings.shape[0]
        self.num_relations = relation_embeddings.shape[0]
        embedding_dim = entity_embeddings.shape[1]
        if relation_embeddings.shape[1] != embedding_dim:
            raise ValueError("Entity and relation embeddings must have the same dimension")

        self.entity_embeddings = nn.Embedding.from_pretrained(entity_embeddings, freeze=True)
        self.relation_embeddings = nn.Embedding.from_pretrained(relation_embeddings, freeze=True)
        self.entity_projection = nn.Linear(embedding_dim, compressed_dim)
        self.relation_projection = nn.Linear(embedding_dim, compressed_dim)
        self.rgcn = RGCNConv(compressed_dim, compressed_dim, num_relations=self.num_relations)
        self.local_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.graph_dropout = nn.Dropout(0.5)
        initial_log_scale = math.log(math.expm1(1.0))
        self.local_bias = nn.Parameter(torch.tensor(0.0))
        self.local_log_scale = nn.Parameter(torch.tensor(initial_log_scale))
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.global_log_scale = nn.Parameter(torch.tensor(initial_log_scale))
        self.register_buffer("local_score_mean", torch.tensor(0.0))
        self.register_buffer("local_score_std", torch.tensor(1.0))
        self.register_buffer("global_score_mean", torch.tensor(0.0))
        self.register_buffer("global_score_std", torch.tensor(1.0))
        self.plausibility_weight = plausibility_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_margin = ranking_margin
        self.score_normalization = score_normalization
        self.compressed_dim = compressed_dim

    def raw_scores(self, graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        subjects, relations, objects = triples.T
        subject_base = self.entity_embeddings(subjects)
        relation_base = self.relation_embeddings(relations)
        object_base = self.entity_embeddings(objects)
        local_input = torch.cat(
            [subject_base, relation_base, object_base, subject_base - object_base,
             subject_base * object_base],
            dim=-1,
        )
        raw_local = self.local_scorer(local_input).squeeze(-1)

        node_ids = torch.arange(self.num_entities, device=triples.device)
        node_features = self.entity_projection(self.entity_embeddings(node_ids))
        graph_embeddings = self.graph_dropout(
            self.rgcn(node_features, graph.edge_index, graph.edge_attr)
        )
        relation_global = self.relation_projection(relation_base)
        global_distance = torch.linalg.vector_norm(
            graph_embeddings[subjects] + relation_global - graph_embeddings[objects],
            ord=1,
            dim=1,
        ) / self.compressed_dim
        raw_global = -global_distance
        return raw_local, raw_global

    @torch.no_grad()
    def update_score_normalization(
        self, graph: Data, triples: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Store deterministic train-score statistics with dropout disabled."""
        was_training = self.training
        self.eval()
        raw_local, raw_global = self.raw_scores(graph, triples)
        self._store_score_normalization(raw_local, raw_global, mask)
        if was_training:
            self.train()

    @torch.no_grad()
    def _store_score_normalization(
        self, raw_local: torch.Tensor, raw_global: torch.Tensor, mask: torch.Tensor
    ) -> None:
        self.local_score_mean.copy_(raw_local[mask].mean())
        self.local_score_std.copy_(raw_local[mask].std(unbiased=False).clamp_min(1e-6))
        self.global_score_mean.copy_(raw_global[mask].mean())
        self.global_score_std.copy_(raw_global[mask].std(unbiased=False).clamp_min(1e-6))

    def calibrate_scores(
        self, raw_local: torch.Tensor, raw_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_normalized = (raw_local - self.local_score_mean) / self.local_score_std
        global_normalized = (raw_global - self.global_score_mean) / self.global_score_std
        local = self.local_bias + functional.softplus(self.local_log_scale) * local_normalized
        global_score = (
            self.global_bias
            + functional.softplus(self.global_log_scale) * global_normalized
        )
        combined = (
            self.plausibility_weight * local
            + (1.0 - self.plausibility_weight) * global_score
        )
        return local, global_score, combined

    def scores(self, graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.calibrate_scores(*self.raw_scores(graph, triples))

    def forward(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, (_, _, combined), _ = self.forward_detailed(graph, triples, labels, mask)
        return loss, combined

    def forward_detailed(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        raw_local, raw_global = self.raw_scores(graph, triples)
        if self.training and self.score_normalization == "dynamic":
            self._store_score_normalization(raw_local, raw_global, mask)
        local, global_score, combined = self.calibrate_scores(raw_local, raw_global)
        masked_labels = labels[mask]
        combined_loss = balanced_bce_with_logits(combined[mask], masked_labels)
        local_loss = balanced_bce_with_logits(local[mask], masked_labels)
        global_loss = balanced_bce_with_logits(global_score[mask], masked_labels)
        ranking_loss = pairwise_ranking_loss(
            combined[mask],
            masked_labels,
            margin=self.ranking_margin,
            randomize_pairs=self.training,
        )
        loss = (
            combined_loss
            + self.auxiliary_loss_weight * (local_loss + global_loss)
            + self.ranking_loss_weight * ranking_loss
        )
        components = {
            "combined": combined_loss,
            "local": local_loss,
            "global": global_loss,
            "ranking": ranking_loss,
        }
        return loss, (local, global_score, combined), components

    def score_frame(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, descriptions: list[str]
    ) -> pd.DataFrame:
        raw_local, raw_global = self.raw_scores(graph, triples)
        local, global_score, combined = self.calibrate_scores(raw_local, raw_global)
        return pd.DataFrame(
            {
                "triple": descriptions,
                "label": labels.detach().cpu().numpy(),
                "raw_local_score": raw_local.detach().cpu().numpy(),
                "raw_global_score": raw_global.detach().cpu().numpy(),
                "local_score": local.detach().cpu().numpy(),
                "global_score": global_score.detach().cpu().numpy(),
                "score": combined.detach().cpu().numpy(),
            }
        )
