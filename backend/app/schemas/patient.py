from typing import Literal

from pydantic import BaseModel, Field


class Patient(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: Literal["male", "female", "other"]
    weight_kg: float = Field(gt=0, le=400)
    symptoms: list[str] = []
    conditions: list[str] = []
