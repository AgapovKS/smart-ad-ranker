"""
Two-Tower model for query ↔ ad semantic matching.
Query tower: fine-tuned sentence encoder (MiniLM)
Ad tower:    MLP over ad categorical features

Training: in-batch negatives contrastive loss (SimCSE-style)
Used for: semantic ad retrieval via FAISS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Optional, Tuple


class QueryEncoder(nn.Module):
    """Fine-tuned sentence encoder for search queries."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, proj_dim: int = 128, freeze_base: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(self.MODEL_NAME)

        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def mean_pool(self, token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_exp = attention_mask.unsqueeze(-1).float()
        return (token_embeds * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def encode(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.encoder(**enc)
        pooled = self.mean_pool(out.last_hidden_state, enc["attention_mask"])
        return F.normalize(self.proj(pooled), dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        return F.normalize(self.proj(pooled), dim=-1)


class AdEncoder(nn.Module):
    """MLP encoder over ad categorical features."""

    def __init__(self, field_dims: list[int], embed_dim: int = 16, proj_dim: int = 128):
        super().__init__()
        total_vocab = sum(field_dims)
        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=0)
        self.register_buffer(
            "offsets",
            torch.tensor([0, *torch.cumsum(torch.tensor(field_dims[:-1]), dim=0).tolist()]),
        )
        in_dim = len(field_dims) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.offsets
        embeds = self.embedding(x).view(x.size(0), -1)
        return F.normalize(self.mlp(embeds), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, query_enc: QueryEncoder, ad_enc: AdEncoder, temperature: float = 0.07):
        super().__init__()
        self.query_enc = query_enc
        self.ad_enc = ad_enc
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        ad_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_emb = self.query_enc(query_input_ids, query_attention_mask)  # (B, D)
        a_emb = self.ad_enc(ad_features)                                # (B, D)
        return q_emb, a_emb

    def contrastive_loss(self, q_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """In-batch negatives NT-Xent loss."""
        B = q_emb.size(0)
        logits = (q_emb @ a_emb.T) / self.temperature.clamp(min=0.01)  # (B, B)
        labels = torch.arange(B, device=q_emb.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss

    def similarity(self, q_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        return (q_emb * a_emb).sum(dim=-1)  # cosine (both normalized)
