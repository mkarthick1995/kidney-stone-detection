from pathlib import Path

from ..schemas.guidance import GuidanceRequest

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")


def build_user_message(req: GuidanceRequest) -> str:
    d = req.detection
    p = req.patient
    detection_line = (
        f"Detection: {'Kidney stone detected' if d.detected else 'No stone detected'}. "
        f"Confidence: {d.confidence:.2f}. "
        f"Location: {d.location}. Size: {d.size_estimate}."
    )
    patient_line = (
        f"Patient: {p.age}-year-old {p.sex}, {p.weight_kg}kg. "
        f"Symptoms: {', '.join(p.symptoms) or 'none reported'}. "
        f"Conditions: {', '.join(p.conditions) or 'none reported'}."
    )
    return (
        f"{patient_line}\n{detection_line}\n\n"
        "Provide structured guidance in the six sections specified in your instructions."
    )
