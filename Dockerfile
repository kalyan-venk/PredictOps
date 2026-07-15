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
FROM python:3.12-slim AS final
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --from=builder /app/src ./src
COPY --from=builder /app/models ./models

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn"]
CMD ["predictops.app:app", "--host", "0.0.0.0", "--port", "8000"]
