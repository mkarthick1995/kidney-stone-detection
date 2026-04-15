"""Class-aligned paired batch sampler for unpaired CT/US datasets.

CT and US datasets are from different patient populations. For fusion training,
we sample same-label batches from both independently and pair them at the
batch level. Document this limitation clearly in the report.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class ClassAlignedPairSampler(Sampler):
    def __init__(self, ct_labels: list[int], us_labels: list[int],
                 batch_size: int = 16, seed: int = 42):
        self.ct_labels = np.array(ct_labels)
        self.us_labels = np.array(us_labels)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        self.ct_by_class = {c: np.where(self.ct_labels == c)[0] for c in np.unique(self.ct_labels)}
        self.us_by_class = {c: np.where(self.us_labels == c)[0] for c in np.unique(self.us_labels)}

    def __iter__(self):
        n_batches = min(len(self.ct_labels), len(self.us_labels)) // self.batch_size
        for _ in range(n_batches):
            # Balanced batch: half positive, half negative
            half = self.batch_size // 2
            ct_idx = np.concatenate([
                self.rng.choice(self.ct_by_class[0], half, replace=False),
                self.rng.choice(self.ct_by_class[1], half, replace=False),
            ])
            us_idx = np.concatenate([
                self.rng.choice(self.us_by_class[0], half, replace=False),
                self.rng.choice(self.us_by_class[1], half, replace=False),
            ])
            perm = self.rng.permutation(self.batch_size)
            yield list(zip(ct_idx[perm].tolist(), us_idx[perm].tolist()))

    def __len__(self) -> int:
        return min(len(self.ct_labels), len(self.us_labels)) // self.batch_size
