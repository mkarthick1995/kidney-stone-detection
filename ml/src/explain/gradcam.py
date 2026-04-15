import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def make_cam(encoder: torch.nn.Module, target_layer) -> GradCAM:
    return GradCAM(model=encoder, target_layers=[target_layer])


def overlay(original_rgb_01: np.ndarray, cam_mask: np.ndarray) -> np.ndarray:
    """original_rgb_01: HxWx3 float [0,1]. cam_mask: HxW float [0,1]. Returns HxWx3 uint8."""
    return show_cam_on_image(original_rgb_01, cam_mask, use_rgb=True)
