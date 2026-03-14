# Embedding Worker — Faculytics

LaBSE embedding worker for the Faculytics analysis pipeline. Receives text via HTTP, returns 768-dimensional L2-normalized embeddings using [sentence-transformers](https://sbert.net/) with ONNX backend for CPU-optimized inference.

## API Contract

### `POST /embeddings`

**Request:**

```json
{
  "jobId": "uuid",
  "version": "1.0",
  "type": "embedding",
  "text": "The professor explains concepts clearly.",
  "metadata": {
    "submissionId": "uuid",
    "facultyId": "faculty-001",
    "versionId": "version-001"
  },
  "publishedAt": "2026-03-14T00:00:00.000Z"
}
```

**Success response** (HTTP 200):

```json
{
  "jobId": "uuid",
  "version": "1.0",
  "status": "completed",
  "result": {
    "embedding": [0.01, 0.02, "... (768 floats)"],
    "modelName": "LaBSE"
  },
  "completedAt": "2026-03-14T00:01:00.000Z"
}
```

**Error response** (HTTP 200 — domain errors avoid BullMQ retries):

```json
{
  "jobId": "uuid",
  "version": "1.0",
  "status": "failed",
  "error": "description",
  "completedAt": "2026-03-14T00:01:00.000Z"
}
```

### `GET /health`

Returns `200 {"status": "ok", "model": "LaBSE"}` when ready, `503` otherwise.

## Quick Start

### Local development

```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn src.main:app --reload

# Run tests
uv run pytest

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Docker

```bash
docker build -t embedding-worker .
docker run -p 8000:8000 embedding-worker
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MODEL_NAME` | `sentence-transformers/LaBSE` | Hugging Face model ID |
| `MODEL_BACKEND` | `onnx` | Inference backend (`onnx` or `torch`) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `OPENAPI_MODE` | `false` | Enable Swagger UI at `/docs` |

Copy `.env.sample` to `.env` to get started.

## Architecture

```
src/
├── config.py       # pydantic-settings configuration
├── models.py       # Pydantic request/response schemas (camelCase aliases)
├── embedding.py    # EmbeddingService: model loading and inference
└── main.py         # FastAPI app, lifespan, routes
```

- **Model loading** happens once at startup via FastAPI's `lifespan` context manager
- **ONNX backend** provides 2-4x faster CPU inference compared to PyTorch
- **Domain errors** return HTTP 200 with `status: "failed"` to prevent BullMQ from retrying bad input — only unexpected server failures return 5xx
- **Contract compliance** — Pydantic schemas use camelCase field aliases matching the Zod schemas in the [NestJS API](https://github.com/CtrlAltElite-Devs/api.faculytics)
