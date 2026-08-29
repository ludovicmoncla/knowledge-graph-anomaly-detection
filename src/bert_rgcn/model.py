from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


class BertRGCN(nn.Module):
    """Combine semantic triple scoring with R-GCN/TransE graph scoring."""

    def __init__(
        self,
        entity_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        compressed_dim: int = 64,
        plausibility_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= plausibility_weight <= 1.0:
            raise ValueError("plausibility_weight must be between 0 and 1")

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
        self.loss_function = nn.BCEWithLogitsLoss()
        self.plausibility_weight = plausibility_weight

    def scores(self, graph: Data, triples: torch.Tensor) -> tuple[torch.Tensor, ...]:
        subjects, relations, objects = triples.T
        subject_base = self.entity_embeddings(subjects)
        relation_base = self.relation_embeddings(relations)
        object_base = self.entity_embeddings(objects)
        local_input = torch.cat(
            [subject_base, relation_base, object_base, subject_base - object_base,
             subject_base * object_base],
            dim=-1,
        )
        local = self.local_scorer(local_input).squeeze(-1)

        node_ids = torch.arange(self.num_entities, device=triples.device)
        node_features = self.entity_projection(self.entity_embeddings(node_ids))
        graph_embeddings = self.graph_dropout(
            self.rgcn(node_features, graph.edge_index, graph.edge_attr)
        )
        relation_global = self.relation_projection(relation_base)
        global_score = -torch.linalg.vector_norm(
            graph_embeddings[subjects] + relation_global - graph_embeddings[objects],
            ord=1,
            dim=1,
        )
        combined = (
            self.plausibility_weight * local
            + (1.0 - self.plausibility_weight) * global_score
        )
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

