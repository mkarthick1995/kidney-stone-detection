"""Lazy-singleton wrapper around the fusion model."""
from functools import lru_cache

import torch

from ..config import settings


@lru_cache(maxsize=1)
def get_model():
    # Import here to avoid pulling torch on module load in tests that mock this
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ml"))
    from src.models.fusion import FusionModel  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(pretrained=False).to(device)
    state = torch.load(settings.model_weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device
