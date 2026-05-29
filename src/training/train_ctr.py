"""
Train DeepFM on processed Avazu data.

Usage:
    python -m src.training.train_ctr \
        --train data/processed/train.parquet \
        --val   data/processed/val.parquet \
        --field_dims data/processed/field_dims.npy \
        --epochs 5 --batch_size 4096 --lr 1e-3 \
        --out checkpoints/deepfm.pt
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss

from src.models.deepfm import DeepFM

CAT_COLS = [
    "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category", "device_id", "device_ip",
    "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
TARGET = "click"


def load_dataset(path: str) -> TensorDataset:
    df = pd.read_parquet(path)
    X = torch.tensor(df[CAT_COLS].values, dtype=torch.long)
    y = torch.tensor(df[TARGET].values, dtype=torch.float)
    return TensorDataset(X, y)


def evaluate(model: DeepFM, loader: DataLoader, device: str) -> dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            p = model.predict_proba(X).cpu().numpy()
            preds.extend(p)
            labels.extend(y.cpu().numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    return {
        "auc": roc_auc_score(labels, preds),
        "logloss": log_loss(labels, preds),
    }


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = load_dataset(args.train)
    val_ds = load_dataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=4)

    field_dims = np.load(args.field_dims).tolist()
    model = DeepFM(field_dims, embed_dim=16, hidden_dims=[400, 400, 400], dropout=0.2).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 500 == 0:
                print(f"  epoch={epoch+1} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["auc"])
        print(f"Epoch {epoch+1}: val_auc={metrics['auc']:.4f} val_logloss={metrics['logloss']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({
                "model_state": model.state_dict(),
                "field_dims": field_dims,
                "val_auc": best_auc,
                "epoch": epoch + 1,
            }, out_path)
            print(f"  -> saved checkpoint (auc={best_auc:.4f})")

    print(f"Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.parquet")
    parser.add_argument("--val", default="data/processed/val.parquet")
    parser.add_argument("--field_dims", default="data/processed/field_dims.npy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="checkpoints/deepfm.pt")
    args = parser.parse_args()
    train(args)
