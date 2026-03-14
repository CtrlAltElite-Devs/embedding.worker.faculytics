from pydantic import BaseModel, ConfigDict, Field


class JobMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    submission_id: str = Field(alias="submissionId")
    faculty_id: str = Field(alias="facultyId")
    version_id: str = Field(alias="versionId")


class EmbeddingRequest(BaseModel):
    """Matches the Zod `analysisJobSchema` from the NestJS API."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "jobId": "550e8400-e29b-41d4-a716-446655440000",
                    "version": "1.0",
                    "type": "embedding",
                    "text": "The professor explains concepts clearly.",
                    "metadata": {
                        "submissionId": "660e8400-e29b-41d4-a716-446655440001",
                        "facultyId": "faculty-001",
                        "versionId": "version-001",
                    },
                    "publishedAt": "2026-03-14T00:00:00.000Z",
                }
            ]
        },
    )

    job_id: str = Field(alias="jobId")
    version: str
    type: str
    text: str = Field(min_length=1)
    metadata: JobMetadata
    published_at: str = Field(alias="publishedAt")


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_name=False)

    embedding: list[float]
    model_name: str = Field(alias="modelName", serialization_alias="modelName")


class EmbeddingSuccessResponse(BaseModel):
    """Matches the Zod `analysisResultSchema` with status='completed'."""

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_name=False,
        json_schema_extra={
            "examples": [
                {
                    "jobId": "550e8400-e29b-41d4-a716-446655440000",
                    "version": "1.0",
                    "status": "completed",
                    "result": {
                        "embedding": [0.1] * 768,
                        "modelName": "LaBSE",
                    },
                    "completedAt": "2026-03-14T00:01:00.000Z",
                }
            ]
        },
    )

    job_id: str = Field(alias="jobId", serialization_alias="jobId")
    version: str
    status: str = "completed"
    result: EmbeddingResult
    completed_at: str = Field(alias="completedAt", serialization_alias="completedAt")


class SimpleEmbeddingRequest(BaseModel):
    """Lightweight request — just a sentence, no job metadata."""

    text: str = Field(min_length=1)


class SimpleEmbeddingResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_name=False,
    )

    embedding: list[float]
    model_name: str = Field(alias="modelName", serialization_alias="modelName")
    completed_at: str = Field(alias="completedAt", serialization_alias="completedAt")


class EmbeddingErrorResponse(BaseModel):
    """Matches the Zod `analysisResultSchema` with status='failed'."""

    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_name=False,
        json_schema_extra={
            "examples": [
                {
                    "jobId": "550e8400-e29b-41d4-a716-446655440000",
                    "version": "1.0",
                    "status": "failed",
                    "error": "Text is empty",
                    "completedAt": "2026-03-14T00:01:00.000Z",
                }
            ]
        },
    )

    job_id: str = Field(alias="jobId", serialization_alias="jobId")
    version: str
    status: str = "failed"
    error: str
    completed_at: str = Field(alias="completedAt", serialization_alias="completedAt")
