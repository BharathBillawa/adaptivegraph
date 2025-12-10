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
        # Supports both sequential (last_context) and async (ID-based) feedback.
        self._last_context: Optional[np.ndarray] = None
        self._last_action: int = -1
        
        # Map event_id -> (context, action_idx)
        self.pending_decisions: Dict[str, Any] = {}
        
        # Map trace_id -> List[(context, action_idx)]
        self.active_traces: Dict[str, List[Any]] = {}

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
        
        # 3b. ID-Based Tracking (if ID present in state)
        if isinstance(state, dict):
            # Check for generic event_id
            event_id = state.get("event_id") or state.get("id") or state.get("run_id")
            if event_id:
                self.pending_decisions[str(event_id)] = (context, action_idx)
            
            # Check for trace_id (Trajectory tracking)
            trace_id = state.get("trace_id")
            if trace_id:
                t_id = str(trace_id)
                if t_id not in self.active_traces:
                    self.active_traces[t_id] = []
                self.active_traces[t_id].append((context, action_idx))
        
        # 4. Return routing decision
        return action_name

    def record_feedback(self, result: Any, reward: Optional[float] = None, event_id: Optional[str] = None) -> None:
        """
        Updates the model based on the result.
        Can be called manually or via a callback hook.
        
        Args:
            result: The result state (used if reward_fn is set).
            reward: Explicit reward float.
            event_id: The ID of the event to provide feedback for (async mode).
        """
        target_context = None
        target_action = -1

        if event_id:
            # Async Mode
            if event_id in self.pending_decisions:
                target_context, target_action = self.pending_decisions.pop(event_id)
            else:
                # Warning: ID not found or already processed
                return
        else:
            # Sequential Mode
            target_context = self._last_context
            target_action = self._last_action
            # Clear pending immediately
            self._last_context = None
            self._last_action = -1

        if target_context is None or target_action == -1:
            return  # No pending action to reward
            
        # Compute reward
        if reward is None:
            if self.reward_fn:
                reward = self.reward_fn(result)
            else:
                reward = 0.0  # Default/Warning?
        
        # Update Policy
        self.policy.update(target_context, target_action, reward)
        
        # Store Experience
        self.memory.add(target_context, target_action, reward)

    def complete_trace(self, trace_id: str, final_reward: float, decay: float = 1.0) -> None:
        """
        Apply a final reward to an entire trajectory of decisions.
        Use this for multi-step agents where intermediate rewards are unknown.
        
        Args:
            trace_id: The identifier for the session/trace.
            final_reward: The success/fail score of the entire trace.
            decay: Discount factor (e.g., 0.9) to apply to earlier steps. 1.0 = equal credit.
        """
        t_id = str(trace_id)
        if t_id not in self.active_traces:
            return
            
        decisions = self.active_traces.pop(t_id)
        current_reward = final_reward
        
        # Iterate in reverse order if applying decay (last step gets full reward, earlier steps get discounted)
        # Or standard credit assignment
        for (ctx, act) in reversed(decisions):
            self.policy.update(ctx, act, current_reward)
            self.memory.add(ctx, act, current_reward)
            current_reward *= decay
