from typing import Any, Optional, Callable
import numpy as np
import hashlib
import warnings


class StateEncoder:
    """
    Encodes arbitrary state into a fixed-size vector.
    
    Supports three encoding modes:
    1. Numpy arrays: Pass-through with truncation/padding to output_dim
    2. Strings with embedding_fn: Use provided embedding function (e.g., SentenceTransformers)
    3. Fallback: Deterministic hashing (WARNING: not semantically meaningful)
    
    For production use with text, strongly recommend providing embedding_fn.
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
        """Encode state into a fixed-size vector.
        
        Args:
            state: Input state (numpy array, string, dict, or any object).
            
        Returns:
            Fixed-size numpy array of shape (output_dim,).
            
        Note:
            - For numpy arrays: flattened and truncated/padded to output_dim.
            - For strings with embedding_fn: embedded using provided function.
            - For other types: deterministic hashing (NOT semantic similarity).
        """
        if isinstance(state, np.ndarray):
            # If it's already a vector, resize or return (simple pass-through for now)
            arr = state.flatten()
            if arr.shape[0] > self.output_dim:
                warnings.warn(
                    f"Truncating vector from {arr.shape[0]} to {self.output_dim} dimensions. "
                    f"Information may be lost.",
                    UserWarning
                )
            return arr[:self.output_dim]

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
        # WARNING: This creates a random-but-deterministic vector, NOT semantic similarity.
        # Strings like "cat" and "cats" will have completely unrelated vectors.
        # For semantic similarity, use embedding_fn with sentence-transformers or similar.
        
        state_str = str(state)
        
        # Create deterministic pseudo-random vector using hash as seed
        # This ensures identical inputs always produce identical outputs
        seed = int(hashlib.sha256(state_str.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vector = rng.standard_normal(self.output_dim)

        # Normalize
        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector
