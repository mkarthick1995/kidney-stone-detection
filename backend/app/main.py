from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import guidance, predict
from .services.model_loader import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager-load model on startup so first /predict is not cold
    get_model()
    yield


app = FastAPI(title="Kidney Stone Detection API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(guidance.router)


@app.get("/health")
def health():
    return {"status": "ok"}
