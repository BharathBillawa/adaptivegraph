import logging
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceStore(Protocol):
    def add(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    def get_all(self) -> Dict[str, Any]: ...


class InMemoryExperienceStore:
    def __init__(self):
        self.contexts: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.metadata: List[Optional[Dict[str, Any]]] = []

    def add(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
                "metadata": [],
            }
        return {
            "contexts": np.stack(self.contexts),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "metadata": self.metadata,
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

    Args:
        dim: Dimension of vectors.
        metric: Distance metric ('cosine' recommended).
        persist_path: Path to save index and metadata (without extension).
        auto_save: If True, save after every add(). If False, call save() manually.
                   Set to False for better performance with frequent updates.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        persist_path: Optional[str] = None,
        auto_save: bool = True,
    ):
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
        self.persist_path = persist_path
        self.auto_save = auto_save

        # Use inner-product index; for cosine, inputs should be normalized
        self.index = faiss.IndexFlatIP(dim)
        self.contexts: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.metadata: List[Optional[Dict[str, Any]]] = []

        if self.persist_path:
            self._load()

    def _load(self):
        import os
        import pickle

        if os.path.exists(self.persist_path + ".index") and os.path.exists(
            self.persist_path + ".pkl"
        ):
            try:
                self.index = self._faiss.read_index(self.persist_path + ".index")
                with open(self.persist_path + ".pkl", "rb") as f:
                    data = pickle.load(f)
                    self.contexts = data["contexts"]
                    self.actions = data["actions"]
                    self.rewards = data["rewards"]
                    self.metadata = data["metadata"]
            except Exception as e:
                logger.warning(
                    f"Failed to load persistence file {self.persist_path}: {e}"
                )

    def save(self):
        if not self.persist_path:
            return
        import pickle

        self._faiss.write_index(self.index, self.persist_path + ".index")
        data = {
            "contexts": self.contexts,
            "actions": self.actions,
            "rewards": self.rewards,
            "metadata": self.metadata,
        }
        with open(self.persist_path + ".pkl", "wb") as f:
            pickle.dump(data, f)

    def add(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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

        if self.persist_path and self.auto_save:
            self.save()

    def get_all(self) -> Dict[str, Any]:
        if not self.contexts:
            return {
                "contexts": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "metadata": [],
            }
        return {
            "contexts": np.stack(self.contexts),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "metadata": self.metadata,
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

        if self.persist_path:
            import os

            if os.path.exists(self.persist_path + ".index"):
                os.remove(self.persist_path + ".index")
            if os.path.exists(self.persist_path + ".pkl"):
                os.remove(self.persist_path + ".pkl")
