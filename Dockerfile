FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .python-version uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ src/

# Download and convert model to ONNX, then remove original PyTorch weights
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/LaBSE', backend='onnx')" \
    && find /root/.cache/huggingface -name "*.bin" -delete \
    && find /root/.cache/huggingface -name "*.safetensors" -delete \
    && rm -rf /root/.cache/uv /root/.cache/pip

# --- Runtime stage ---
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH"
ENV HOST=0.0.0.0
ENV PORT=5201

EXPOSE 5201

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5201"]
