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
