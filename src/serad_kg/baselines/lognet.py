from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


def build_line_graph_edges(triples: np.ndarray, graph_mask: np.ndarray) -> torch.Tensor:
    """Build LoGNet's line graph without allocating the original dense n x n matrix."""
    subject_to_triples: dict[int, list[int]] = defaultdict(list)
    object_to_triples: dict[int, list[int]] = defaultdict(list)
    graph_indices = np.flatnonzero(graph_mask)
    for index in graph_indices:
        subject, _, object_ = triples[index]
        subject_to_triples[int(subject)].append(int(index))
        object_to_triples[int(object_)].append(int(index))
    edges: set[tuple[int, int]] = set()
    for index in graph_indices:
        subject, _, object_ = triples[index]
        neighbors = (
            subject_to_triples[int(object_)]
            + object_to_triples[int(subject)]
            + subject_to_triples[int(subject)]
            + object_to_triples[int(object_)]
        )
        edges.update((int(index), neighbor) for neighbor in neighbors if neighbor != index)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.as_tensor(sorted(edges), dtype=torch.long).T.contiguous()


class LoGNet(nn.Module):
    """Scalable implementation of the LoGNet comparison baseline."""

    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        line_graph_edge_index: torch.Tensor,
        embedding_dim: int = 16,
        plausibility_weight: float = 0.3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.plausibility_weight = plausibility_weight
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.bigru = nn.GRU(embedding_dim, embedding_dim, batch_first=True, bidirectional=True)
        self.project = nn.Linear((6 * embedding_dim) + 2, embedding_dim)
        self.gat = GATConv(embedding_dim, embedding_dim, heads=1, dropout=dropout)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.register_buffer("line_graph_edge_index", line_graph_edge_index)
        uniform_range = 6 / np.sqrt(embedding_dim)
        self.entity_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        self.relation_embeddings.weight.data.uniform_(-uniform_range, uniform_range)
        self.relation_embeddings.weight.data.div_(
            self.relation_embeddings.weight.data.norm(p=1, dim=1, keepdim=True).clamp_min(1e-12)
        )

    def scores(self, _graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        subjects, relations, objects = triples.T
        vectorized = torch.stack(
            [
                self.entity_embeddings(subjects),
                self.relation_embeddings(relations),
                self.entity_embeddings(objects),
            ],
            dim=1,
        )
        recurrent, _ = self.bigru(vectorized)
        subject_repr = recurrent[:, 0, :]
        relation_repr = recurrent[:, 1, :]
        object_repr = recurrent[:, 2, :]
        similarity_so = functional.cosine_similarity(subject_repr, object_repr, dim=1)[:, None]
        similarity_sr = functional.cosine_similarity(subject_repr, relation_repr, dim=1)[:, None]
        local = -torch.linalg.vector_norm(subject_repr + relation_repr - object_repr, dim=1)
        concatenated = torch.cat(
            [subject_repr, relation_repr, object_repr, similarity_so, similarity_sr], dim=1
        )
        projected = self.project(functional.normalize(concatenated, p=2, dim=1))
        graph_embeddings = self.gat(projected, self.line_graph_edge_index)
        global_score = -torch.linalg.vector_norm(graph_embeddings, dim=1) - local
        combined = self.plausibility_weight * local + global_score
        return local, global_score, combined

    def forward(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, combined = self.scores(graph, triples)
        return self.loss_function(combined[mask], labels[mask]), combined

    def score_frame(
        self, graph: Data, triples: torch.Tensor, labels: torch.Tensor, descriptions: list[str]
    ) -> pd.DataFrame:
        local, global_score, combined = self.scores(graph, triples)
        return pd.DataFrame(
            {
                "triple": descriptions,
                "label": labels.detach().cpu().numpy(),
                "local_score": local.detach().cpu().numpy(),
                "global_score": global_score.detach().cpu().numpy(),
                "score": combined.detach().cpu().numpy(),
            }
        )
