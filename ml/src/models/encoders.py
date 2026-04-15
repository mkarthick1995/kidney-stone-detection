import timm
import torch.nn as nn


def build_efficientnet_b4(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """EfficientNet-B4 baseline for single-modality classification."""
    return timm.create_model(
        "tf_efficientnet_b4",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_feature_encoder(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 as a feature extractor (global pooled, 1792-d)."""
    return timm.create_model(
        "tf_efficientnet_b4",
        pretrained=pretrained,
        num_classes=0,  # drop classifier
        global_pool="avg",
    )
