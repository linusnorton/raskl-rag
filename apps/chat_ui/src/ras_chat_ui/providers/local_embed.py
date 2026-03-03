"""Local embedding provider via sentence-transformers."""

from __future__ import annotations

import logging
import math

from .base import EmbedProvider

log = logging.getLogger(__name__)

_model = None


def _truncate_and_normalize(embedding: list[float], dimensions: int) -> list[float]:
    vec = embedding[:dimensions]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


class LocalEmbedProvider(EmbedProvider):
    def __init__(self, model_path: str, device: str, dimensions: int, task_prefix: str):
        self.model_path = model_path
        self.device = device
        self.dimensions = dimensions
        self.task_prefix = task_prefix

    def _get_model(self):
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s on %s...", self.model_path, self.device)
            _model = SentenceTransformer(self.model_path, device=self.device, truncate_dim=self.dimensions)
            log.info("Embedding model loaded.")
        return _model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        prefixed = [f"{self.task_prefix}{t}" for t in texts]
        embeddings = model.encode(prefixed, normalize_embeddings=True)

        result = []
        for emb in embeddings:
            vec = emb.tolist()
            if len(vec) > self.dimensions:
                vec = _truncate_and_normalize(vec, self.dimensions)
            result.append(vec)
        return result
