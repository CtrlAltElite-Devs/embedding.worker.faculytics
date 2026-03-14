import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_name: str = ""

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self, model_name: str, backend: str = "onnx") -> None:
        logger.info("Loading model %s with backend=%s ...", model_name, backend)
        self._model = SentenceTransformer(model_name, backend=backend)
        self._model_name = model_name.split("/")[-1]  # e.g. "LaBSE"
        logger.info("Model loaded: %s", self._model_name)

    def encode(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
