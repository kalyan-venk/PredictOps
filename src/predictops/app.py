import json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request

from predictops.config import MODEL_META_PATH, ONNX_MODEL_PATH
from predictops.schemas import InfoResponse, PredictRequest, PredictResponse


def _load_session() -> ort.InferenceSession | None:
    if not ONNX_MODEL_PATH.exists():
        return None
    return ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])


def _load_meta() -> dict[str, Any]:
    if MODEL_META_PATH.exists():
        return json.loads(MODEL_META_PATH.read_text())
    return {}


def _build_feed(session: ort.InferenceSession, record: dict[str, Any]) -> dict[str, np.ndarray]:
    """One typed 1x1 tensor per model input: string columns as object arrays,
    everything else as float32 (matches the initial_types used at export)."""
    feed: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        value = record[inp.name]
        if "string" in inp.type:
            feed[inp.name] = np.array([[str(value)]], dtype=object)
        else:
            feed[inp.name] = np.array([[float(value)]], dtype=np.float32)
    return feed


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = _load_session()
    app.state.meta = _load_meta()
    yield


app = FastAPI(title="PredictOps Serving API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness probe target for load balancers and orchestrators."""
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    """Report whether a model is loaded and which classifier the ONNX graph came from."""
    session = request.app.state.session
    meta = request.app.state.meta
    return InfoResponse(
        loaded=session is not None,
        classifier_type=meta.get("classifier_type", "none") if session else "none",
        artifact_path=str(ONNX_MODEL_PATH),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score a single record and return the predicted class and probability."""
    session = request.app.state.session
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    try:
        outputs = session.run(None, _build_feed(session, payload.model_dump()))
        prediction = int(np.asarray(outputs[0]).ravel()[0])
        probability = float(np.asarray(outputs[1])[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    return PredictResponse(prediction=prediction, probability=probability)


@app.post("/reload")
def reload_model(request: Request) -> dict[str, str]:
    """Reload the model artifact from disk without restarting the process."""
    request.app.state.session = _load_session()
    request.app.state.meta = _load_meta()
    if request.app.state.session is None:
        raise HTTPException(status_code=503, detail="Model file not found; cannot reload.")
    return {"status": "reloaded"}
