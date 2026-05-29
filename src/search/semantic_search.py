"""
Semantic ad retrieval using FAISS.
Builds an index over ad embeddings from AdEncoder,
then retrieves top-K ads for a given query string.
"""
import numpy as np
import torch
import faiss
import pickle
from pathlib import Path
from typing import Optional

from src.models.two_tower import QueryEncoder, AdEncoder, TwoTowerModel


class SemanticAdSearch:
    def __init__(
        self,
        model: TwoTowerModel,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.index: Optional[faiss.Index] = None
        self.meta: Optional[list] = None  # list of dicts per ad

        if index_path and index_path.exists():
            self.load_index(index_path, meta_path)

    # ------------------------------------------------------------------ build
    def build_index(
        self,
        ad_features: np.ndarray,       # (N, num_fields) int64
        ad_meta: list[dict],            # N dicts with human-readable info
        batch_size: int = 1024,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> None:
        all_embs = []
        with torch.no_grad():
            for start in range(0, len(ad_features), batch_size):
                chunk = torch.tensor(ad_features[start:start + batch_size], dtype=torch.long).to(self.device)
                emb = self.model.ad_enc(chunk).cpu().numpy()
                all_embs.append(emb)
                if start % 50000 == 0:
                    print(f"  indexed {start}/{len(ad_features)}")

        embeddings = np.vstack(all_embs).astype("float32")
        dim = embeddings.shape[1]

        # Inner product index (vectors are L2-normalized → cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.meta = ad_meta
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

        if index_path:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))
            with open(meta_path, "wb") as f:
                pickle.dump(self.meta, f)
            print(f"Saved index -> {index_path}")

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        with torch.no_grad():
            q_emb = self.model.query_enc.encode([query], device=self.device)
            q_np = q_emb.cpu().numpy().astype("float32")

        scores, indices = self.index.search(q_np, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.meta[idx])
            entry["similarity"] = float(score)
            results.append(entry)
        return results

    def batch_search(self, queries: list[str], top_k: int = 5) -> list[list[dict]]:
        with torch.no_grad():
            q_emb = self.model.query_enc.encode(queries, device=self.device)
            q_np = q_emb.cpu().numpy().astype("float32")

        scores, indices = self.index.search(q_np, top_k)
        out = []
        for i in range(len(queries)):
            results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx >= 0:
                    entry = dict(self.meta[idx])
                    entry["similarity"] = float(score)
                    results.append(entry)
            out.append(results)
        return out

    # --------------------------------------------------------------- persist
    def load_index(self, index_path: Path, meta_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")
