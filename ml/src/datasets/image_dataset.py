from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageDataset(Dataset):
    """Generic single-modality dataset driven by a manifest CSV.

    Manifest columns: path, label, source, patient_id
    """

    def __init__(self, manifest_path: str | Path, transform=None, size: int = 512):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["path"]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return {"image": img, "label": int(row["label"]), "path": str(row["path"])}
