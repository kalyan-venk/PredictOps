from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from predictops.config import MODEL_PATH
from predictops.schemas import InfoResponse, PredictRequest, PredictResponse


def _load_model() -> Any | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _load_model()
    yield


app = FastAPI(title="PredictOps Serving API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe target for load balancers and orchestrators."""
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    """Report whether a model is loaded and which classifier/artifact it is."""
    model = request.app.state.model
    return InfoResponse(
        loaded=model is not None,
        classifier_type=type(model.named_steps["classifier"]).__name__ if model else "none",
        artifact_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score a single record and return the predicted class and probability."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    try:
        row = pd.DataFrame([payload.model_dump()])
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return PredictResponse(prediction=prediction, probability=probability)


@app.post("/reload")
def reload_model(request: Request) -> dict[str, str]:
    """Reload the model artifact from disk without restarting the process."""
    request.app.state.model = _load_model()
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model file not found; cannot reload.")
    return {"status": "reloaded"}
