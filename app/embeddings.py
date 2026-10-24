from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Devuelve un vector float32 normalizado (norma L2 = 1)."""
        ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class FastEmbedEmbedder:
    """Embeddings locales vía ONNX (fastembed): sin GPU ni llamadas de red tras la descarga inicial."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(self.embed("warmup"))

    def embed(self, text: str) -> np.ndarray:
        return _normalize(next(iter(self._model.embed([text]))))
