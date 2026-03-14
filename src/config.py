from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 5201
    model_name: str = "sentence-transformers/LaBSE"
    model_backend: str = "onnx"
    log_level: str = "INFO"
    openapi_mode: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
