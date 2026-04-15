#!/usr/bin/env bash
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json
set -euo pipefail

RAW=data/raw
mkdir -p "$RAW"/{ct,ct_external,us}

echo "Downloading CT (Nazmul Islam, Bangladesh)..."
kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone \
    -p "$RAW/ct" --unzip

echo "Downloading US (Gurjeetkaurmangat)..."
kaggle datasets download -d gurjeetkaurmangat/kidney-ultrasound-images-stone-and-no-stone \
    -p "$RAW/us" --unzip

echo "Downloading external CT (Abdalla 2025 Iraq) — update slug if Kaggle mirror differs"
# kaggle datasets download -d <abdalla-iraq-slug> -p "$RAW/ct_external" --unzip

echo "Done. Raw data in $RAW/"
