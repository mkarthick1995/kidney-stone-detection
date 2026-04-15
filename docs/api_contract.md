# API Contract — v1.0

Freeze this before building the backend or frontend. Changes require bumping version and updating both Pydantic schemas and TypeScript types.

## POST /predict

Multipart form upload.

| Field | Type | Required | Notes |
|---|---|---|---|
| `ct_image` | file (PNG/JPG) | yes | max 10MB |
| `us_image` | file (PNG/JPG) | yes | max 10MB |
| `patient` | JSON string | yes | see below |

### `patient` schema
```json
{
  "age": 35,
  "sex": "male",
  "weight_kg": 72,
  "symptoms": ["flank_pain", "hematuria"],
  "conditions": ["hypertension"]
}
```

### Response 200
```json
{
  "detected": true,
  "confidence": 0.91,
  "location": "right_renal_pelvis",
  "size_estimate": "small_lt_5mm",
  "ct_heatmap_b64": "iVBORw0KGgo...",
  "us_heatmap_b64": "iVBORw0KGgo...",
  "model_version": "fusion-v1.0",
  "inference_ms": 840
}
```

### Error responses
- `422` — validation error (missing image, oversized file, bad patient JSON)
- `500` — model failure (includes `error_id` for log correlation)
- `503` — model not loaded (during startup)

---

## POST /guidance

Request body (JSON):
```json
{
  "detection": { /* full /predict response */ },
  "patient": { /* same patient object */ }
}
```

Response: `text/event-stream` (SSE).

### Event types
| Event | Data | Order |
|---|---|---|
| `section_start` | `{"section": "diet"}` | once per section |
| `token` | `{"text": "Drink..."}` | many per section |
| `section_end` | `{"section": "diet"}` | once per section |
| `error` | `{"message": "..."}` | terminal |
| `done` | `{}` | terminal, after all sections |

### Section order (fixed)
1. `diet`
2. `hydration`
3. `exercise`
4. `otc_medicine`
5. `red_flags`
6. `disclaimer` (mandatory — if missing, retry once then error)

### Error responses
- `422` — invalid detection/patient payload
- `502` — Claude API failure
- `504` — Claude API timeout (>30s)
