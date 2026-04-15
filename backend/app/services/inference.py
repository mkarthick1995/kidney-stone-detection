import base64
import io
import time

import cv2
import numpy as np
import torch
from PIL import Image

from .model_loader import get_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_and_normalize(image_bytes: bytes, size: int) -> tuple[np.ndarray, torch.Tensor]:
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    img = cv2.resize(img, (size, size))
    rgb01 = img.astype(np.float32) / 255.0
    tensor = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).contiguous()
    return rgb01, tensor


def _to_b64_png(arr_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def run_inference(ct_bytes: bytes, us_bytes: bytes) -> dict:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model, device = get_model()
    ct_rgb, ct_tensor = _load_and_normalize(ct_bytes, 512)
    us_rgb, us_tensor = _load_and_normalize(us_bytes, 224)
    ct_tensor, us_tensor = ct_tensor.to(device), us_tensor.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(ct_tensor, us_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ct_target = model.ct_encoder.blocks[-1]
    us_target = model.us_encoder.blocks[-1]
    ct_cam = GradCAM(model=model.ct_encoder, target_layers=[ct_target])
    us_cam = GradCAM(model=model.us_encoder, target_layers=[us_target])
    ct_mask = ct_cam(input_tensor=ct_tensor)[0]
    us_mask = us_cam(input_tensor=us_tensor)[0]

    ct_overlay = show_cam_on_image(ct_rgb, ct_mask, use_rgb=True)
    us_overlay = show_cam_on_image(us_rgb, us_mask, use_rgb=True)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    stone_prob = float(probs[1])
    return {
        "detected": stone_prob >= 0.5,
        "confidence": stone_prob if stone_prob >= 0.5 else 1.0 - stone_prob,
        "location": "unknown",  # TODO: derive from Grad-CAM centroid
        "size_estimate": "unknown",
        "ct_heatmap_b64": _to_b64_png(ct_overlay),
        "us_heatmap_b64": _to_b64_png(us_overlay),
        "model_version": "fusion-v1.0",
        "inference_ms": elapsed_ms,
    }
