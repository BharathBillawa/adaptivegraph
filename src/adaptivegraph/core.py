from typing import Any, Callable, List, Optional, Dict
import numpy as np
from .encoder import StateEncoder
from .embedding import SentenceTransformerEmbedding
from .policy import LinUCBPolicy
from .memory import InMemoryExperienceStore, ExperienceStore

class LearnableEdge:
    @classmethod
    @classmethod
    def create(
        cls,
        options: List[str],
        embedding: str = "sentence-transformers",
        memory: str = "faiss",
        memory_persist_path: Optional[str] = None,
        feature_dim: int = 32,
        **kwargs
    ) -> "LearnableEdge":
        """
        Factory method to create a LearnableEdge with pre-configured components.
        
        Args:
            options: List of available actions.
            embedding: Encoding strategy ("sentence-transformers", etc).
            memory: Experience storage strategy ("faiss", "memory").
            memory_persist_path: Path prefix for saving memory (if memory="faiss").
            feature_dim: Dimension of state vector.
            **kwargs: Additional args passed to LearnableEdge constructor.
        """
        # 1. Setup Embedding
        embedding_fn = None
        if embedding == "sentence-transformers":
            from .embedding import SentenceTransformerEmbedding
            embedding_fn = SentenceTransformerEmbedding(dim=feature_dim)
        else:
            raise ValueError(f"Unknown embedding option: {embedding}")

        # 2. Setup Memory
        experience_store = None
        if memory == "faiss":
            from .memory import FaissExperienceStore
            experience_store = FaissExperienceStore(dim=feature_dim, persist_path=memory_persist_path)
        elif memory == "memory":
             experience_store = InMemoryExperienceStore()
        else:
             raise ValueError(f"Unknown memory option: {memory}")

        return cls(
            options=options,
            feature_dim=feature_dim,
            embedding_fn=embedding_fn,
            experience_store=experience_store,
            **kwargs
        )

    def __init__(
        self,
        options: List[str],
        reward_fn: Optional[Callable[[Any], float]] = None,
        policy: str = "linucb",
        feature_dim: int = 32,
        embedding_fn: Optional[Callable[[str], Any]] = None,
        encoder_normalize: bool = True,
        experience_store: Optional[ExperienceStore] = None,
        exploration_alpha: float = 1.0,
        value_key: Optional[str] = None,
    ):
        self.options = options
        self.reward_fn = reward_fn
        self.value_key = value_key
        
        # Components
        self.encoder = StateEncoder(output_dim=feature_dim, embedding_fn=embedding_fn, normalize=encoder_normalize)

        # Memory
        self.memory = experience_store if experience_store is not None else InMemoryExperienceStore()
        
        if policy == "linucb":
            self.policy = LinUCBPolicy(n_actions=len(options), feature_dim=feature_dim, alpha=exploration_alpha)
        else:
            raise ValueError(f"Unknown policy: {policy}")
            
        # State tracking for feedback
        # NOTE: This is simplistic and assumes sequential execution or single-request scope.
        # In a real async server, this needs a request_id/thread_local.
        self._last_context: Optional[np.ndarray] = None
        self._last_action: int = -1

    def __call__(self, state: Any) -> str:
        """
        The main routing entry point.
        """
        # 1. Extract Value (if key provided)
        value_to_encode = state
        if self.value_key:
            if isinstance(state, dict):
                value_to_encode = state.get(self.value_key, state)
            elif hasattr(state, self.value_key):
                value_to_encode = getattr(state, self.value_key)
                
        # 2. Encode
        context = self.encoder.encode(value_to_encode)
        
        # 2. Select Action
        action_idx = self.policy.select_action(context)
        action_name = self.options[action_idx]
        
        # 3. Store temporary state for feedback
        self._last_context = context
        self._last_action = action_idx
        
        # 4. Return routing decision
        return action_name

    def record_feedback(self, result: Any, reward: Optional[float] = None) -> None:
        """
        Updates the model based on the result.
        Can be called manually or via a callback hook.
        """
        if self._last_context is None or self._last_action == -1:
            return  # No pending action to reward
            
        # Compute reward
        if reward is None:
            if self.reward_fn:
                reward = self.reward_fn(result)
            else:
                reward = 0.0  # Default/Warning?
        
        # Update Policy
        self.policy.update(self._last_context, self._last_action, reward)
        
        # Store Experience
        self.memory.add(self._last_context, self._last_action, reward)
        
        # Clear pending
        self._last_context = None
        self._last_action = -1
