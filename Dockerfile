# ---------- builder: install deps + train the model ----------
FROM python:3.12-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/src
COPY src/ src/
COPY data/ data/
RUN python -m predictops.train

# ---------- final: only what's needed to serve ----------
# The model is served from ONNX via onnxruntime, so the runtime image carries
# none of the training/scientific stack (scikit-learn, xgboost, pandas, scipy,
# mlflow, evidently) — just onnxruntime + numpy + the FastAPI layer.
FROM python:3.12-slim AS final
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt \
 && SP=/usr/local/lib/python3.12/site-packages \
 && find $SP -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true \
 && find $SP -type d -name tests -prune -exec rm -rf {} + 2>/dev/null || true \
 && find $SP -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true \
 && pip uninstall -y pip setuptools wheel 2>/dev/null || true

COPY --from=builder /app/src ./src
COPY --from=builder /app/models/model.onnx ./models/model.onnx
COPY --from=builder /app/models/model_meta.json ./models/model_meta.json

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["predictops.app:app", "--host", "0.0.0.0", "--port", "8000"]
