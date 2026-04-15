"""Preprocess raw datasets into (image, label) manifest CSVs.

Usage:
    python scripts/preprocess.py --modality ct
    python scripts/preprocess.py --modality us
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def window_ct_hu(img: np.ndarray, level: int = 40, width: int = 400) -> np.ndarray:
    """Apply HU windowing. Input is already rescaled 8-bit; this is a no-op placeholder
    for when DICOM pixel arrays are the source. Replace with pydicom-based pipeline
    when DICOM files are used directly."""
    return img


def preprocess_ct(raw_dir: Path, out_dir: Path, size: int = 512) -> pd.DataFrame:
    """Expects raw_dir with subfolders 'Stone' and 'Normal' (or similar)."""
    rows = []
    for label_name, label in [("Stone", 1), ("Normal", 0)]:
        for img_path in (raw_dir / label_name).rglob("*.[jJpP]*[gG]"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = window_ct_hu(img)
            img = cv2.resize(img, (size, size))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out_path = out_dir / label_name / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img_rgb)
            rows.append({
                "path": str(out_path),
                "label": label,
                "source": "ct_bangladesh",
                "patient_id": img_path.stem,
            })
    return pd.DataFrame(rows)


def preprocess_us(raw_dir: Path, out_dir: Path, size: int = 224) -> pd.DataFrame:
    rows = []
    for label_name, label in [("Stone", 1), ("Normal", 0)]:
        for img_path in (raw_dir / label_name).rglob("*.[jJpP]*[gG]"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            img = cv2.resize(img, (size, size))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out_path = out_dir / label_name / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img_rgb)
            rows.append({
                "path": str(out_path),
                "label": label,
                "source": "us_kaggle",
                "patient_id": img_path.stem,
            })
    return pd.DataFrame(rows)


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int = 42):
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "us"], required=True)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.modality == "ct":
        raw = Path(args.raw or "data/raw/ct")
        out = Path(args.out or "data/processed/ct")
        df = preprocess_ct(raw, out)
    else:
        raw = Path(args.raw or "data/raw/us")
        out = Path(args.out or "data/processed/us")
        df = preprocess_us(raw, out)

    out.mkdir(parents=True, exist_ok=True)
    split_and_save(df, out)


if __name__ == "__main__":
    main()
