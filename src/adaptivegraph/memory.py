from typing import Protocol, List, Dict, Any, Optional
import numpy as np

class ExperienceStore(Protocol):
    def add(self, context: np.ndarray, action: int, reward: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        ...
    
    def get_all(self) -> Dict[str, Any]:
        ...

class InMemoryExperienceStore:
    def __init__(self):
        self.contexts: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.metadata: List[Optional[Dict[str, Any]]] = []

    def add(self, context: np.ndarray, action: int, reward: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)
        self.metadata.append(metadata)

    def get_all(self) -> Dict[str, Any]:
        # Return stacked arrays for easier processing
        if not self.contexts:
            return {
                "contexts": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "metadata": []
            }
        return {
            "contexts": np.stack(self.contexts),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "metadata": self.metadata
        }

    def clear(self):
        self.contexts = []
        self.actions = []
        self.rewards = []
        self.metadata = []


class FaissExperienceStore:
    """
    Simple local FAISS-backed store. Keeps vectors in an index and mirrors
    actions/rewards/metadata in Python lists.

    Requirements: `faiss-cpu` package.
    Metric: cosine similarity implemented via normalized vectors and inner product (IP).
    """
    def __init__(self, dim: int, metric: str = "cosine"):
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError(
                "faiss-cpu package is required for FaissExperienceStore.\n"
                "Install with: pip install 'faiss-cpu'"
            ) from e

        self.dim = dim
        self.metric = metric
        self._faiss = faiss

        # Use inner-product index; for cosine, inputs should be normalized
        self.index = faiss.IndexFlatIP(dim)
        self.contexts: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.metadata: List[Optional[Dict[str, Any]]] = []

    def add(self, context: np.ndarray, action: int, reward: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Normalize for cosine if needed
        vec = context.astype(np.float32)
        if self.metric == "cosine":
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        self.index.add(vec.reshape(1, -1))
        self.contexts.append(vec)
        self.actions.append(action)
        self.rewards.append(reward)
        self.metadata.append(metadata)

    def get_all(self) -> Dict[str, Any]:
        if not self.contexts:
            return {
                "contexts": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "metadata": []
            }
        return {
            "contexts": np.stack(self.contexts),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "metadata": self.metadata
        }

    def query_similar(self, context: np.ndarray, k: int = 5) -> Dict[str, Any]:
        vec = context.astype(np.float32)
        if self.metric == "cosine":
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
        if len(self.contexts) == 0:
            return {"indices": np.array([]), "scores": np.array([])}
        distances, indices = self.index.search(vec.reshape(1, -1), k)
        return {"indices": indices[0], "scores": distances[0]}

    def clear(self):
        self.index.reset()
        self.contexts = []
        self.actions = []
        self.rewards = []
        self.metadata = []
