from typing import Literal

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    detected: bool
    confidence: float = Field(ge=0, le=1)
    location: str
    size_estimate: Literal["small_lt_5mm", "medium_5_10mm", "large_gt_10mm", "unknown"]
    ct_heatmap_b64: str
    us_heatmap_b64: str
    model_version: str
    inference_ms: int
