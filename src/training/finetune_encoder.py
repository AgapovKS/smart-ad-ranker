"""
Fine-tune QueryEncoder + AdEncoder on synthetic click-based contrastive pairs.

Strategy:
  - Positive pair: (query, ad) where the ad was clicked
  - In-batch negatives: all other ads in the batch
  - Synth queries: generated from ad category / site_category via a template

Usage:
    python -m src.training.finetune_encoder \
        --data data/processed/train.parquet \
        --field_dims data/processed/field_dims.npy \
        --epochs 5 --batch_size 256 --lr 3e-5 \
        --out checkpoints/two_tower.pt
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from src.models.two_tower import QueryEncoder, AdEncoder, TwoTowerModel

# Synthetic query generation from category codes
CATEGORY_QUERIES = {
    "0f2161f8": ["buy shoes online", "footwear shop", "sneakers sale"],
    "f028772b": ["smartphone deals", "mobile phone buy", "best phone 2024"],
    "50e219e0": ["travel booking", "cheap flights", "hotel deals"],
    "28905ebd": ["gaming laptop", "best gpu", "pc games"],
    "04d2e4c7": ["food delivery", "order pizza", "restaurant nearby"],
    "__default__": ["best deals today", "shop online", "discount offers"],
}


def synth_query(app_category: str) -> str:
    queries = CATEGORY_QUERIES.get(str(app_category), CATEGORY_QUERIES["__default__"])
    return np.random.choice(queries)


class ClickPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cat_cols: list[str]):
        # Only clicked impressions → positive pairs
        clicked = df[df["click"] == 1].reset_index(drop=True)
        self.queries = [synth_query(row["app_category"]) for _, row in clicked.iterrows()]
        self.ad_features = clicked[cat_cols].values.astype(np.int64)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.ad_features[idx]


def collate_fn(batch):
    queries, features = zip(*batch)
    return list(queries), torch.tensor(np.stack(features), dtype=torch.long)


CAT_COLS = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category", "device_id", "device_ip",
    "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_parquet(args.data)
    field_dims = np.load(args.field_dims).tolist()

    dataset = ClickPairDataset(df, CAT_COLS)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    query_enc = QueryEncoder(proj_dim=128, freeze_base=False)
    ad_enc = AdEncoder(field_dims, embed_dim=16, proj_dim=128)
    model = TwoTowerModel(query_enc, ad_enc).to(device)

    # Separate LRs: lower for pretrained encoder
    optimizer = AdamW([
        {"params": model.query_enc.encoder.parameters(), "lr": args.lr * 0.1},
        {"params": model.query_enc.proj.parameters(), "lr": args.lr},
        {"params": model.ad_enc.parameters(), "lr": args.lr},
        {"params": [model.temperature], "lr": 1e-3},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (queries, ad_feats) in enumerate(loader):
            ad_feats = ad_feats.to(device)

            # Tokenize queries
            enc = model.query_enc.tokenizer(
                queries, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            q_emb, a_emb = model(enc["input_ids"], enc["attention_mask"], ad_feats)
            loss = model.contrastive_loss(q_emb, a_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"  epoch={epoch+1} step={step}/{len(loader)} loss={loss.item():.4f} "
                      f"temp={model.temperature.item():.3f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.parquet")
    parser.add_argument("--field_dims", default="data/processed/field_dims.npy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--out", default="checkpoints/two_tower.pt")
    args = parser.parse_args()
    train(args)
