"""Fusion model trainer. Two-stage: (A) frozen encoders, (B) fine-tune last blocks.
    python -m src.training.train_fusion --config configs/fusion.yaml --stage A
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.image_dataset import ImageDataset
from ..datasets.paired_sampler import ClassAlignedPairSampler
from ..models.fusion import FusionModel
from ..utils.config import load_config
from ..utils.seed import set_seed


class PairedLoader:
    """Wraps two single-modality datasets + a pair sampler."""

    def __init__(self, ct_ds, us_ds, sampler):
        self.ct_ds, self.us_ds, self.sampler = ct_ds, us_ds, sampler

    def __iter__(self):
        for batch_pairs in self.sampler:
            ct_imgs, us_imgs, labels = [], [], []
            for ct_i, us_i in batch_pairs:
                ct_item = self.ct_ds[ct_i]
                us_item = self.us_ds[us_i]
                # By construction the sampler draws same-class pairs
                assert ct_item["label"] == us_item["label"]
                ct_imgs.append(ct_item["image"])
                us_imgs.append(us_item["image"])
                labels.append(ct_item["label"])
            yield {
                "ct": torch.stack(ct_imgs),
                "us": torch.stack(us_imgs),
                "label": torch.tensor(labels),
            }

    def __len__(self):
        return len(self.sampler)


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if train else "val"):
            ct = batch["ct"].to(device)
            us = batch["us"].to(device)
            labels = batch["label"].to(device)
            if train:
                optimizer.zero_grad()
            logits = model(ct, us)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * ct.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += ct.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ct_train = ImageDataset(cfg["data"]["ct_train"], size=512)
    us_train = ImageDataset(cfg["data"]["us_train"], size=224)
    ct_val = ImageDataset(cfg["data"]["ct_val"], size=512)
    us_val = ImageDataset(cfg["data"]["us_val"], size=224)

    train_sampler = ClassAlignedPairSampler(
        ct_train.df["label"].tolist(), us_train.df["label"].tolist(),
        batch_size=cfg["batch_size"],
    )
    val_sampler = ClassAlignedPairSampler(
        ct_val.df["label"].tolist(), us_val.df["label"].tolist(),
        batch_size=cfg["batch_size"], seed=123,
    )
    train_loader = PairedLoader(ct_train, us_train, train_sampler)
    val_loader = PairedLoader(ct_val, us_val, val_sampler)

    model = FusionModel(pretrained=True).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    if args.stage == "A":
        model.freeze_encoders()
        lr = cfg["stage_a"]["lr"]
        epochs = cfg["stage_a"]["epochs"]
    else:
        model.unfreeze_last_blocks(n=2)
        lr = cfg["stage_b"]["lr"]
        epochs = cfg["stage_b"]["epochs"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cfg.get("class_weights", [1.0, 1.0]), device=device)
    )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, device, False)
        scheduler.step()
        print(f"[stage {args.stage}] epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / f"fusion_stage_{args.stage}_best.pt")


if __name__ == "__main__":
    main()
