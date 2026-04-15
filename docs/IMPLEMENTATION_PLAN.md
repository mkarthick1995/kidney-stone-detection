# Kidney Stone Detection & Clinical Guidance — Implementation Plan

A week-by-week, file-by-file execution plan derived from `kidney_stone_full_plan_v2.docx`. Work top to bottom; nothing in a later phase blocks on anything undecided in an earlier one.

---

## 0. Repository layout

Create one monorepo with three top-level folders. Keep ML, backend, and frontend independent so they can be developed in parallel once the API contract is frozen (end of week 2).

```
kidney-stone-system/
├── ml/                         # training + model code
│   ├── data/                   # raw/, interim/, processed/ (gitignored)
│   ├── notebooks/              # EDA, debugging only — no training runs
│   ├── src/
│   │   ├── datasets/           # CTDataset, USDataset, PairedDataset
│   │   ├── models/             # encoders.py, fusion.py, classifier.py
│   │   ├── training/           # train_baseline.py, train_fusion.py
│   │   ├── evaluation/         # metrics.py, delong.py, ablation.py
│   │   ├── explain/            # gradcam.py
│   │   └── utils/              # config.py, seed.py, io.py
│   ├── configs/                # yaml configs per experiment
│   ├── scripts/                # download_data.sh, preprocess.py, export_onnx.py
│   ├── artifacts/              # weights, heatmaps, logs (gitignored)
│   └── requirements.txt
├── backend/                    # FastAPI inference + LLM
│   ├── app/
│   │   ├── main.py             # FastAPI app
│   │   ├── routers/            # predict.py, guidance.py
│   │   ├── services/           # model_loader.py, gradcam_service.py, claude_service.py
│   │   ├── schemas/            # Pydantic request/response models
│   │   ├── prompts/            # system_prompt.md, guidance_template.py
│   │   └── config.py
│   ├── tests/
│   └── requirements.txt
├── frontend/                   # React + Vite + Tailwind
│   ├── src/
│   │   ├── pages/              # UploadPage, ResultsPage
│   │   ├── components/         # Dropzone, PatientForm, HeatmapView, GuidancePanel
│   │   ├── hooks/              # usePredict.ts, useGuidanceStream.ts
│   │   ├── api/                # client.ts (axios + EventSource wrapper)
│   │   └── types/              # shared API types (mirrors backend schemas)
│   └── package.json
├── docs/                       # api_contract.md, ethics.md, report/
├── .github/workflows/          # ci.yml (lint + tests)
└── README.md
```

---

## 1. API contract — freeze this first (Week 1, day 1–2)

Everyone blocks on this. Write `docs/api_contract.md` before any feature work.

### POST /predict (multipart)
**Request:** `ct_image: file`, `us_image: file`, `patient: JSON string` (age, sex, weight_kg, symptoms[], conditions[])
**Response:**
```json
{
  "detected": true,
  "confidence": 0.91,
  "location": "right_renal_pelvis",
  "size_estimate": "small_lt_5mm",
  "ct_heatmap_b64": "...",
  "us_heatmap_b64": "...",
  "model_version": "fusion-v1.0",
  "inference_ms": 840
}
```

### POST /guidance (SSE)
**Request:** detection result + patient JSON.
**Response:** `text/event-stream` with events `section_start`, `token`, `section_end`, `done`. Sections are fixed: `diet`, `hydration`, `exercise`, `otc_medicine`, `red_flags`, `disclaimer`.

Define Pydantic schemas in `backend/app/schemas/` and mirror them as TypeScript types in `frontend/src/types/` — keep these in sync manually (or generate with `datamodel-code-generator` → `openapi-typescript`).

---

## 2. Phase-by-phase execution

### Phase 1 — Data (Weeks 1–2)

**Deliverable:** `ml/data/processed/{ct,us}/{train,val,test}/` with PNGs and a `manifest.csv` per split (`path, label, source, patient_id`).

1. `scripts/download_data.sh` — pulls 3 Kaggle datasets via `kaggle datasets download`. Commit the script, not the data.
2. `scripts/preprocess.py` — one entrypoint, CLI flags `--modality ct|us`. For CT: DICOM→PNG via `pydicom`, HU window W:400 L:40, resize 512×512. For US: bilateral filter (OpenCV `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)`), resize 224×224. Normalize to ImageNet stats at dataloader time, not on disk.
3. Stratified 70/15/15 split with `sklearn.model_selection.StratifiedGroupKFold` keyed on `patient_id` where available — prevents patient leakage across splits. Seed = 42, write the split to `splits.json`.
4. `src/datasets/` — `CTDataset`, `USDataset` (each returns `{image, label}`), and `PairedBatchSampler` that yields same-class (stone/no-stone) batches drawn independently from CT and US pools. Document clearly: these are **unpaired** datasets; fusion trains on class-aligned synthetic pairs.
5. EDA notebook (`notebooks/01_eda.ipynb`): class balance, image size distribution, sample visualizations. Keep it short — do not let it bloat.

**Gate to Phase 2:** manifest CSVs exist, dataloaders return correctly-shaped tensors, a 16-image sanity batch visualizes correctly.

### Phase 2 — Single-modality baselines (Weeks 3–4)

**Deliverable:** two trained checkpoints + a results table.

1. `src/models/encoders.py` — `build_efficientnet_b4(pretrained=True, num_classes=2)` via `timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=2)`.
2. `src/training/train_baseline.py` — config-driven (YAML). AdamW, lr=3e-4, cosine schedule, class-weighted BCE, early stopping on val AUC (patience=7). Log to W&B.
3. Augmentations via Albumentations: `Rotate(±15)`, `HorizontalFlip`, `RandomBrightnessContrast`, `ElasticTransform` (CT only; US is already noisy).
4. Train CT baseline and US baseline. Record sensitivity, specificity, AUC, F1 on test set. These numbers are what the fusion model must beat — if it doesn't, the project has no narrative.

**Gate to Phase 3:** both baselines converge above 0.80 AUC. If not, fix data quality before building fusion.

### Phase 3 — Fusion model (Weeks 5–7)

**Deliverable:** `FusionModel` class, Grad-CAM working for both branches.

1. `src/models/fusion.py`:
   - `ProjectionHead`: `Linear(1792, 512) → ReLU → LayerNorm(512)`
   - `CrossAttentionFusion`: two `nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)` — one for CT-attends-to-US, one reversed. Concatenate outputs, `Linear(1024, 512)`, residual.
   - `Classifier`: `Linear(512, 128) → BatchNorm → ReLU → Dropout(0.4) → Linear(128, 2)`.
2. `src/models/fusion_model.py` wraps encoders + projections + fusion + classifier. Forward takes `(ct, us)` pair.
3. `src/training/train_fusion.py`:
   - **Stage A (Weeks 5–6):** freeze both EfficientNet backbones. Train projection + fusion + classifier. AdamW, lr=1e-3, 30 epochs.
   - **Stage B (Week 7):** unfreeze last 2 MBConv blocks per encoder. Fine-tune end-to-end at lr=1e-5, 15 epochs. Gradient checkpointing (`torch.utils.checkpoint`) if VRAM < 16GB.
4. `src/explain/gradcam.py` — use `pytorch-grad-cam`. Target layer: last conv block of each encoder (`encoder.blocks[-1]`). Return heatmaps as numpy arrays; encoding to base64 happens in the backend, not here.

**Gate to Phase 4:** fusion model beats both baselines on val AUC. If cross-attention underperforms concat fusion, keep both in the ablation — the negative result is still publishable.

### Phase 4 — Evaluation & ablation (Weeks 8–9)

**Deliverable:** `results/ablation_table.md`, external validation numbers, DeLong p-values.

1. `src/evaluation/metrics.py` — sensitivity, specificity, AUC, F1, confusion matrix, bootstrap 95% CIs (1000 resamples).
2. `src/evaluation/ablation.py` — runs all four configs (CT-only, US-only, concat, cross-attn) on the same test set, emits markdown table.
3. `src/evaluation/delong.py` — DeLong's test for AUC comparison. Use the `delong` package or port the Sun & Xu 2014 implementation.
4. External validation: run the final fusion model on Abdalla 2025 Iraq CT dataset **without retraining**. CT-only mode (no paired US available) — this demonstrates the CT encoder generalizes. Report as a separate row.

**Gate to Phase 5:** final checkpoint and metrics frozen. Export weights to `artifacts/fusion_v1.pt`. No further training — any "small tweak" from here is a trap.

### Phase 5 — Backend (Weeks 10–11)

**Deliverable:** both endpoints working against the frozen weights.

1. `app/services/model_loader.py` — singleton; loads weights at FastAPI `@app.on_event("startup")`. Keep model in `eval()` mode, move to CUDA if available, else CPU.
2. `app/routers/predict.py` — accepts multipart, validates via Pydantic, runs inference, generates Grad-CAM overlays, encodes to base64 PNG, returns response. Timeout: 10s.
3. `app/services/claude_service.py` — wraps Anthropic SDK. Model: `claude-sonnet-4-6`. Use streaming (`client.messages.stream(...)`). **Enable prompt caching** on the system prompt + few-shot examples — that block is identical across every request, so caching cuts latency and cost dramatically.
4. `app/prompts/system_prompt.md` — medical advisor persona, evidence-based framing, hard disclaimers, structured output contract (the 5 sections). Check this into git.
5. `app/routers/guidance.py` — builds prompt from request, streams Claude response, re-emits as SSE with section boundary events so the frontend can render per-section.
6. `tests/` — pytest with `httpx.AsyncClient`. Mock the Claude client for `/guidance` tests; use a tiny fixture model for `/predict` tests so CI stays fast.

**Env vars:** `ANTHROPIC_API_KEY`, `MODEL_WEIGHTS_PATH`, `ALLOWED_ORIGINS`. Load via `pydantic-settings`.

### Phase 6 — Frontend (Weeks 12–13)

**Deliverable:** full UX flow working against the real backend.

1. Scaffold: `npm create vite@latest frontend -- --template react-ts`, add Tailwind, axios.
2. `UploadPage`: two `Dropzone` components (react-dropzone), `PatientForm` with zod validation. Submit triggers `/predict`.
3. `ResultsPage`: four-panel grid — CT original, CT heatmap, US original, US heatmap. Detection badge (green/red), confidence bar, location chip.
4. `GuidancePanel`: opens `EventSource` against `/guidance`. Five labelled section cards populate in order. Show a persistent disclaimer banner at top — not dismissible.
5. `useGuidanceStream` hook — manages SSE lifecycle, handles reconnection, parses section events.
6. Loading states everywhere. Errors: distinguish network, validation, model, and LLM errors — different messaging.

**Gate to Phase 7:** upload → result → guidance works end-to-end on localhost.

### Phase 7 — Prompt tuning (Week 14)

**Deliverable:** `docs/prompt_evaluation.md` with 20+ test cases.

1. Build a test matrix: `{detected, not_detected} × {small, medium, large stone} × {patient variants}` = ~20 scenarios.
2. For each, run the prompt and grade output on: structure adherence, clinical appropriateness, safety framing, consistency across runs.
3. Iterate system prompt until all 20 pass. Lock the prompt; treat further changes like a code review.
4. Add a content filter: if Claude returns anything without the disclaimer section, retry once, then fail loud.

### Phase 8 — Integration & polish (Week 15)

1. E2E test script: `scripts/e2e_test.py` — uploads a known image pair, checks response shape, streams guidance, validates all 5 sections arrive.
2. Edge cases: missing modality (return 422), very low confidence (<0.5 → UI shows "inconclusive"), Claude API timeout (show cached fallback message), oversized images (reject >10MB).
3. Record a demo video (`docs/demo.mp4`) — this is worth more in the final defense than any extra metric point.

### Phase 9 — Report (Weeks 16–17)

Standard structure: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion (including failure modes — don't hide them), Ethics, Conclusion. Include ablation table, external validation, and screenshots. ~40 pages.

---

## 3. Critical path & parallelization

- **Solo developer:** strict linear phases — total 17 weeks as planned.
- **Two people:** once the API contract is frozen (end of week 2), person A owns ML through week 9, person B starts backend scaffolding (week 3) with a mock model, then frontend (week 6) against the mocked backend. Real integration happens week 10 when weights are ready. Saves ~4 weeks.

---

## 4. Biggest risks to watch

1. **Unpaired datasets.** The fusion is trained on synthetic class-aligned pairs, not true patient pairs. Be upfront about this in the report — it's the first thing an examiner will ask.
2. **Overfitting cross-attention.** 8 heads over 512 dims on a small dataset overfits fast. Watch val AUC vs train AUC gap; if it diverges past 0.05, reduce heads to 4.
3. **External validation failure.** If the Iraq dataset scores drop >10 points, it reveals the model learned dataset-specific artifacts, not stones. Don't hide this — frame it as a finding and discuss domain shift.
4. **LLM giving medical advice.** The disclaimer is not optional and must be in the system prompt, every response, and the UI. If it's missing anywhere, the ethics section will fail.
5. **Scope creep.** You will be tempted to add segmentation, size regression, or a mobile app. Don't. Ship the v2 plan.

---

## 5. First-week checklist

- [ ] Create repo with the layout above
- [ ] Write and commit `docs/api_contract.md`
- [ ] Set up Kaggle API credentials and download all 3 datasets
- [ ] Set up W&B project and `.env.example`
- [ ] Run `scripts/preprocess.py` end-to-end on a 100-image subset
- [ ] Train EfficientNet-B4 for 1 epoch on CT just to verify the pipeline
- [ ] Open GitHub issues for every deliverable in Phases 1–4

Start with the pipeline sanity run, not the data cleaning. If the pipeline is broken, cleaner data won't save it.
