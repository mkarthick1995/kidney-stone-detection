from pydantic import BaseModel

from .patient import Patient
from .predict import PredictResponse


class GuidanceRequest(BaseModel):
    detection: PredictResponse
    patient: Patient
