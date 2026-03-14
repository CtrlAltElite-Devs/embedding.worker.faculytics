import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import settings
from src.embedding import EmbeddingService
from src.models import (
    EmbeddingErrorResponse,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSuccessResponse,
    SimpleEmbeddingRequest,
    SimpleEmbeddingResponse,
)

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    embedding_service.load(settings.model_name, settings.model_backend)
    yield


app = FastAPI(
    title="Faculytics Embedding Worker",
    version="0.1.0",
    description="LaBSE embedding worker for the Faculytics analysis pipeline",
    lifespan=lifespan,
    openapi_url="/openapi.json" if settings.openapi_mode else None,
)


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    if not embedding_service.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model": None},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model": embedding_service.model_name},
    )


@app.post(
    "/embeddings",
    response_model=EmbeddingSuccessResponse,
    responses={200: {"description": "Embedding result (completed or failed)"}},
    tags=["embeddings"],
)
async def create_embedding(request: EmbeddingRequest) -> JSONResponse:
    now = datetime.now(UTC).isoformat()

    try:
        vector = embedding_service.encode(request.text)
    except Exception as exc:
        logger.exception("Embedding failed for job %s", request.job_id)
        error_response = EmbeddingErrorResponse(
            jobId=request.job_id,
            version=request.version,
            error=str(exc),
            completedAt=now,
        )
        return JSONResponse(
            status_code=200,
            content=error_response.model_dump(by_alias=True),
        )

    success_response = EmbeddingSuccessResponse(
        jobId=request.job_id,
        version=request.version,
        result=EmbeddingResult(embedding=vector, modelName=embedding_service.model_name),
        completedAt=now,
    )
    return JSONResponse(
        status_code=200,
        content=success_response.model_dump(by_alias=True),
    )


@app.post(
    "/embed",
    response_model=SimpleEmbeddingResponse,
    tags=["embeddings"],
)
async def embed_text(request: SimpleEmbeddingRequest) -> JSONResponse:
    now = datetime.now(UTC).isoformat()
    vector = embedding_service.encode(request.text)
    response = SimpleEmbeddingResponse(
        embedding=vector,
        modelName=embedding_service.model_name,
        completedAt=now,
    )
    return JSONResponse(
        status_code=200,
        content=response.model_dump(by_alias=True),
    )
