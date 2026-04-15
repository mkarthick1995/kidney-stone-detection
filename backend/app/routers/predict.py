import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..schemas.patient import Patient
from ..schemas.predict import PredictResponse
from ..services.inference import run_inference

router = APIRouter(tags=["predict"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/predict", response_model=PredictResponse)
async def predict(
    ct_image: UploadFile = File(...),
    us_image: UploadFile = File(...),
    patient: str = Form(...),
):
    try:
        Patient.model_validate_json(patient)
    except Exception as e:
        raise HTTPException(422, f"Invalid patient JSON: {e}")

    ct_bytes = await ct_image.read()
    us_bytes = await us_image.read()
    if len(ct_bytes) > MAX_IMAGE_BYTES or len(us_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(422, "Image exceeds 10MB limit")

    try:
        result = run_inference(ct_bytes, us_bytes)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    return PredictResponse(**result)
