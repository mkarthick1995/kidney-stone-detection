# Kidney Stone Detection & Clinical Guidance System

Multi-modal deep learning (CT + ultrasound) with LLM-powered clinical guidance.

## Structure
- `ml/` — PyTorch training, fusion model, evaluation
- `backend/` — FastAPI inference + Claude guidance (SSE)
- `frontend/` — React + Vite + Tailwind UI
- `docs/` — API contract, ethics, report

## Quick start

```bash
# ML
cd ml && pip install -r requirements.txt
bash scripts/download_data.sh
python scripts/preprocess.py --modality ct
python -m src.training.train_baseline --config configs/baseline_ct.yaml

# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env  # fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

See `IMPLEMENTATION_PLAN.md` (repo root) and `docs/api_contract.md` for details.
