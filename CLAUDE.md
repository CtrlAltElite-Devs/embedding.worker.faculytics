# CLAUDE.md

## Project Overview

LaBSE embedding worker for the Faculytics analysis pipeline. Receives text via HTTP POST, produces 768-dimensional L2-normalized embeddings using `sentence-transformers` with ONNX backend.

## Main Caller

The NestJS API at `https://github.com/CtrlAltElite-Devs/api.faculytics` (develop branch) dispatches embedding jobs via BullMQ. The `EmbeddingProcessor` POSTs to this worker.

**Contract schemas in the API repo:**
- `src/modules/analysis/dto/analysis-job-message.dto.ts` — request schema (Zod)
- `src/modules/analysis/dto/analysis-result-message.dto.ts` — response schema (Zod)
- `src/modules/analysis/processors/base.processor.ts` — HTTP dispatch logic
- `src/modules/analysis/processors/embedding.processor.ts` — extracts `result.embedding` and `result.modelName`

## Common Commands

```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn src.main:app --reload

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Docker
docker build -t embedding-worker .
docker run -p 8000:8000 embedding-worker
```

## Architecture

- **Framework**: FastAPI with Pydantic validation
- **Inference**: `sentence-transformers` with ONNX backend for CPU-optimized inference
- **Model**: `sentence-transformers/LaBSE` (768-dim, L2-normalized)
- **Error strategy**: Domain errors return HTTP 200 with `status: "failed"` (avoids BullMQ retries). Only unexpected failures return HTTP 5xx.

### File Structure

```
src/
├── config.py       # pydantic-settings: host, port, model_name, model_backend, log_level, openapi_mode
├── models.py       # Pydantic request/response schemas (camelCase aliases to match Zod)
├── embedding.py    # EmbeddingService: load model, encode text
└── main.py         # FastAPI app, lifespan, /embeddings + /health routes
```

## Configuration

Environment variables (see `.env.sample`):

- `HOST` (default `0.0.0.0`), `PORT` (default `8000`) — server binding
- `MODEL_NAME` (default `sentence-transformers/LaBSE`), `MODEL_BACKEND` (default `onnx`)
- `LOG_LEVEL` (default `INFO`)
- `OPENAPI_MODE` (default `false`) — enable Swagger UI at `/docs`
