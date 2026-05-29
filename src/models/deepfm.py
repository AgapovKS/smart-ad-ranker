"""
DeepFM for CTR prediction.
Paper: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
       Guo et al., IJCAI 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FMLayer(nn.Module):
    """Second-order feature interactions via FM."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_fields, embed_dim)
        square_of_sum = x.sum(dim=1) ** 2                     # (B, E)
        sum_of_square = (x ** 2).sum(dim=1)                   # (B, E)
        return 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)  # (B, 1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFM(nn.Module):
    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        hidden_dims: List[int] = (400, 400, 400),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim

        # Shared embedding table (one row per unique value across all fields)
        total_vocab = sum(field_dims)
        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=0)

        # Field offsets for embedding lookup
        self.register_buffer(
            "offsets",
            torch.tensor([0, *torch.cumsum(torch.tensor(field_dims[:-1]), dim=0).tolist()]),
        )

        # FM part
        self.fm = FMLayer()

        # Deep part
        self.mlp = MLP(self.num_fields * embed_dim, list(hidden_dims), dropout)

        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_fields) — integer indices
        x = x + self.offsets  # shift indices to global vocab
        embeds = self.embedding(x)  # (B, F, E)

        fm_out = self.fm(embeds)                     # (B, 1)
        deep_out = self.mlp(embeds.view(embeds.size(0), -1))  # (B, 1)
        logit = fm_out + deep_out + self.bias
        return logit.squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class IsotonicCalibrator(nn.Module):
    """
    Platt scaling calibration on top of DeepFM.
    Trains a (a, b) pair: P_calibrated = sigmoid(a * logit + b)
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)
