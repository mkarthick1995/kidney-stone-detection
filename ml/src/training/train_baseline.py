"""Single-modality baseline trainer. Usage:
    python -m src.training.train_baseline --config configs/baseline_ct.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.image_dataset import ImageDataset
from ..models.encoders import build_efficientnet_b4
from ..utils.config import load_config
from ..utils.seed import set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train"):
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ImageDataset(cfg["data"]["train_manifest"], size=cfg["data"]["size"])
    val_ds = ImageDataset(cfg["data"]["val_manifest"], size=cfg["data"]["size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    model = build_efficientnet_b4(pretrained=True, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    class_weights = torch.tensor(cfg.get("class_weights", [1.0, 1.0]), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    main()
