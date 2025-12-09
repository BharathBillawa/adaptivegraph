from typing import Any, Optional, Callable
import numpy as np
import hashlib

class StateEncoder:
    """
    Encodes arbitrary state into a fixed-size vector.
    v1: Supports explicit vector pass-through or a simple deterministic hashing for strings/dicts.
    """
    def __init__(
        self,
        output_dim: int = 32,
        embedding_fn: Optional[Callable[[str], Any]] = None,
        normalize: bool = True,
    ):
        self.output_dim = output_dim
        self.embedding_fn = embedding_fn
        self.normalize = normalize

    def encode(self, state: Any) -> np.ndarray:
        if isinstance(state, np.ndarray):
            # If it's already a vector, resize or return (simple pass-through for now)
            return state.flatten()[:self.output_dim]  # Naive truncation/flat

        if self.embedding_fn and isinstance(state, str):
            # Use provided embedding function
            # Expecting it to return list or array
            vec = self.embedding_fn(state)
            arr = np.asarray(vec, dtype=np.float32).flatten()
            # Enforce output_dim by truncation/padding
            if arr.shape[0] >= self.output_dim:
                arr = arr[:self.output_dim]
            else:
                out = np.zeros(self.output_dim, dtype=np.float32)
                out[:arr.shape[0]] = arr
                arr = out
            if self.normalize:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            return arr

        # Fallback: Deterministic hashing for string/dict representation
        # This is a 'poor man's embedding' for MVP without heavy ML deps
        # It projects text features into a random fixed space (SimHash-ish)

        state_str = str(state)
        # Seeded random projection based on hash
        # Use sha256 of string to seed a generator?
        # Or simpler: just hash to buckets.

        # Let's do a simple hashed feature vector
        vector = np.zeros(self.output_dim)
        # Hash the string multiple times with different salts to simulate dimensions?
        # Or just use the bytes.

        # Simple consistent hashing approach:
        # Hash string -> int -> seed.
        # Generate 'output_dim' random numbers.

        seed = int(hashlib.sha256(state_str.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.standard_normal(self.output_dim)

        # Normalize
        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector
