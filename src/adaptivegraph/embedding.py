from typing import Any

import numpy as np


class SentenceTransformerEmbedding:
    """
    Lightweight wrapper for Sentence Transformers that returns a vector shaped
    to the requested dimension.

    Notes:
    - Requires `sentence-transformers` to be installed.
    - Uses `normalize_embeddings=True` for stable bandit behavior.
    - Truncates or zero-pads to `dim`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "sentence-transformers package is required for SentenceTransformerEmbedding.\n"
                "Install with: pip install 'sentence-transformers'"
            ) from e
        self.model = SentenceTransformer(model_name)
        self.dim = dim

    def __call__(self, text: str) -> Any:
        vec = self.model.encode(text, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32).flatten()
        if arr.shape[0] >= self.dim:
            return arr[: self.dim]
        out = np.zeros(self.dim, dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
