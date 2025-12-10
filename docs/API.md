# API Reference

## Core Classes

### LearnableEdge

The main class for creating learnable conditional edges.

#### Constructor

```python
LearnableEdge(
    options: List[str],
    reward_fn: Optional[Callable[[Any], float]] = None,
    policy: str = "linucb",
    feature_dim: int = 32,
    embedding_fn: Optional[Callable[[str], Any]] = None,
    encoder_normalize: bool = True,
    experience_store: Optional[ExperienceStore] = None,
    exploration_alpha: float = 1.0,
    value_key: Optional[str] = None,
)
```

**Parameters:**
- `options` (List[str]): List of possible actions/routes. Must be non-empty and contain unique values.
- `reward_fn` (Optional[Callable]): Function to compute reward from result state. If None, reward must be provided explicitly.
- `policy` (str): Bandit policy to use. Currently supports: `"linucb"`.
- `feature_dim` (int): Dimension of state vectors. Must be positive. Default: 32.
- `embedding_fn` (Optional[Callable]): Function to embed strings into vectors. If None, uses deterministic hashing.
- `encoder_normalize` (bool): Whether to normalize encoded vectors. Default: True.
- `experience_store` (Optional[ExperienceStore]): Storage backend for experiences. If None, uses InMemoryExperienceStore.
- `exploration_alpha` (float): Exploration parameter for LinUCB. Higher values = more exploration. Must be non-negative. Default: 1.0.
- `value_key` (Optional[str]): If state is a dict, extract this key for encoding.

**Raises:**
- `ValueError`: If options is empty, contains duplicates, feature_dim <= 0, or exploration_alpha < 0.

#### Methods

##### `__call__(state: Any) -> str`

Select an action based on the current state.

```python
action = edge(state)
```

**Parameters:**
- `state`: Input state (string, dict, numpy array, or any object).

**Returns:**
- `str`: Selected action name from `options`.

**Notes:**
- For dicts with `event_id`, `id`, or `run_id`, stores decision for async feedback.
- For dicts with `trace_id`, adds decision to trajectory for trace-level rewards.

##### `record_feedback(result: Any, reward: Optional[float] = None, event_id: Optional[str] = None) -> None`

Update the model with feedback about the last decision.

```python
edge.record_feedback(result={}, reward=1.0)
# OR with event_id for async feedback
edge.record_feedback(result={}, reward=1.0, event_id="req_123")
```

**Parameters:**
- `result`: Result state after action execution.
- `reward` (Optional[float]): Explicit reward value. If None, uses `reward_fn(result)`.
- `event_id` (Optional[str]): For async feedback, the ID of the decision to reward.

**Raises:**
- `ValueError`: If reward is NaN or infinite.

**Notes:**
- In sequential mode (no event_id), rewards the last action taken via `__call__`.
- In async mode (with event_id), rewards a specific past decision.
- State is cleared after feedback to prevent double-rewarding.

##### `complete_trace(trace_id: str, final_reward: float, decay: float = 1.0) -> None`

Apply a final reward to an entire trajectory of decisions.

```python
edge.complete_trace(trace_id="session_123", final_reward=1.0, decay=0.9)
```

**Parameters:**
- `trace_id` (str): Identifier for the trajectory/session.
- `final_reward` (float): Success/failure score for the entire trajectory.
- `decay` (float): Discount factor for earlier steps. 1.0 = equal credit, <1.0 = later steps get more credit. Default: 1.0.

**Raises:**
- `ValueError`: If final_reward is NaN or infinite.

**Notes:**
- Useful for multi-step agents where intermediate rewards are unknown.
- Rewards are applied in reverse order (last step first).
- Trace is removed from active_traces after completion.

##### `save_policy(path: str) -> None`

Save the learned policy state to disk.

```python
edge.save_policy("models/my_policy")
```

**Parameters:**
- `path` (str): File path without extension (adds `.pkl` automatically).

**Notes:**
- Saves A and b matrices, n_actions, feature_dim, and alpha.
- Required for persistence across application restarts.

##### `load_policy(path: str) -> None`

Load policy state from disk.

```python
edge.load_policy("models/my_policy")
```

**Parameters:**
- `path` (str): File path without extension.

**Raises:**
- `FileNotFoundError`: If policy file doesn't exist.
- `ValueError`: If loaded policy doesn't match current configuration (n_actions or feature_dim mismatch).

#### Class Method

##### `create(...) -> LearnableEdge`

Factory method for creating edges with pre-configured components.

```python
edge = LearnableEdge.create(
    options=["A", "B"],
    embedding="sentence-transformers",
    memory="faiss",
    memory_persist_path="./data/memory",
    feature_dim=384
)
```

**Parameters:**
- `options` (List[str]): List of possible actions.
- `embedding` (str): Embedding strategy. Options: `"sentence-transformers"`.
- `memory` (str): Storage backend. Options: `"memory"`, `"faiss"`.
- `memory_persist_path` (Optional[str]): Path for persistent storage (faiss only).
- `feature_dim` (int): Dimension of state vectors.
- `**kwargs`: Additional arguments passed to LearnableEdge constructor.

**Raises:**
- `ValueError`: If embedding or memory option is unknown.
- `ImportError`: If required packages are not installed.

---

## Encoders

### StateEncoder

Encodes arbitrary state into fixed-size vectors.

```python
encoder = StateEncoder(
    output_dim=32,
    embedding_fn=None,
    normalize=True
)
```

**Parameters:**
- `output_dim` (int): Size of output vectors.
- `embedding_fn` (Optional[Callable]): Function to embed strings. If None, uses deterministic hashing.
- `normalize` (bool): Whether to L2-normalize output vectors.

**Methods:**

##### `encode(state: Any) -> np.ndarray`

Encode state into a vector.

**Encoding Strategy:**
1. **Numpy arrays**: Pass-through with truncation/padding to output_dim
2. **Strings with embedding_fn**: Use provided embedding function
3. **Fallback**: Deterministic hashing (NOT semantically meaningful)

**Warning:** The fallback hashing creates random-but-deterministic vectors. "cat" and "cats" will be completely unrelated. For semantic similarity, use embedding_fn with sentence-transformers.

---

## Policies

### LinUCBPolicy

Linear Upper Confidence Bound policy with disjoint linear models.

```python
policy = LinUCBPolicy(
    n_actions=3,
    feature_dim=32,
    alpha=1.0,
    ridge_lambda=1.0
)
```

**Parameters:**
- `n_actions` (int): Number of possible actions.
- `feature_dim` (int): Dimension of context vectors.
- `alpha` (float): Exploration parameter. Higher = more exploration.
- `ridge_lambda` (float): Regularization parameter for ridge regression.

**Methods:**

##### `select_action(context: np.ndarray) -> int`

Select action using UCB formula.

**Formula:** `p_a = θ_a^T * x + α * sqrt(x^T * A_a^-1 * x)`

where `θ_a = A_a^-1 * b_a`

##### `update(context: np.ndarray, action: int, reward: float) -> None`

Update policy with observed reward.

**Updates:**
- `A_a += x * x^T`
- `b_a += r * x`

---

## Memory Stores

### InMemoryExperienceStore

Simple in-memory storage for experiences.

```python
store = InMemoryExperienceStore()
```

**Methods:**

##### `add(context: np.ndarray, action: int, reward: float, metadata: Optional[Dict] = None) -> None`

Add an experience.

##### `get_all() -> Dict[str, Any]`

Retrieve all experiences as stacked arrays.

**Returns:**
```python
{
    "contexts": np.ndarray,  # shape (N, feature_dim)
    "actions": np.ndarray,   # shape (N,)
    "rewards": np.ndarray,   # shape (N,)
    "metadata": List[Optional[Dict]]
}
```

##### `clear() -> None`

Remove all experiences.

### FaissExperienceStore

FAISS-backed storage with persistence and similarity search.

```python
store = FaissExperienceStore(
    dim=32,
    metric="cosine",
    persist_path="./data/experiences",
    auto_save=True
)
```

**Parameters:**
- `dim` (int): Dimension of vectors.
- `metric` (str): Distance metric. Options: `"cosine"` (recommended).
- `persist_path` (Optional[str]): Path to save index and metadata.
- `auto_save` (bool): If True, save after every add(). Set to False for batch operations.

**Additional Methods:**

##### `save() -> None`

Manually save index and metadata to disk.

##### `query_similar(context: np.ndarray, k: int = 5) -> Dict[str, Any]`

Find k most similar experiences.

**Returns:**
```python
{
    "indices": np.ndarray,  # Indices of similar experiences
    "scores": np.ndarray    # Similarity scores
}
```

---

## Reward Scorers

### ErrorScorer

Assigns rewards based on error presence.

```python
scorer = ErrorScorer(
    error_keys=["error", "exception"],
    penalty=-1.0,
    success_reward=1.0
)
```

**Usage:**
```python
edge = LearnableEdge(options=["A", "B"], reward_fn=scorer.score)
```

### LLMScorer

Uses LLM-as-a-judge for reward calculation.

```python
scorer = LLMScorer(
    llm_callable=my_llm,
    prompt_template="Rate this response from 0-1: {response}"
)
```

**Requirements:**
- `llm_callable` must accept string prompt and return string response
- Supports LangChain Runnables with `.invoke()`

---

## Best Practices

### 1. Always Provide Feedback

```python
# ❌ Bad: No learning happens
action = edge(state)

# ✅ Good: Model learns
action = edge(state)
result = execute_action(action)
edge.record_feedback(result, reward=calculate_reward(result))
```

### 2. Persist Policy State

```python
# Save periodically
if iteration % 100 == 0:
    edge.save_policy("checkpoints/policy_latest")

# Load on startup
try:
    edge.load_policy("checkpoints/policy_latest")
except FileNotFoundError:
    pass  # Start fresh
```

### 3. Use Semantic Embeddings for Text

```python
# ❌ Bad: Deterministic hashing has no semantic meaning
edge = LearnableEdge(options=["A", "B"])

# ✅ Good: Semantic similarity preserved
edge = LearnableEdge.create(
    options=["A", "B"],
    embedding="sentence-transformers",
    feature_dim=384
)
```

### 4. Tune Exploration

```python
# High exploration (early training)
edge = LearnableEdge(options=["A", "B"], exploration_alpha=2.0)

# Low exploration (exploitation phase)
edge = LearnableEdge(options=["A", "B"], exploration_alpha=0.1)
```

### 5. Batch Memory Operations

```python
# For FaissExperienceStore
store = FaissExperienceStore(dim=32, auto_save=False)
# ... add many experiences ...
store.save()  # Save once at the end
```

---

## Thread Safety

**Warning:** LearnableEdge is **not thread-safe** for sequential feedback mode.

For concurrent environments:
1. Use `event_id` for all feedback
2. OR create one LearnableEdge instance per request
3. OR use external synchronization (locks)

```python
# ✅ Safe: Async feedback with event_id
edge({"query": "...", "event_id": request_id})
# ... later, possibly different thread ...
edge.record_feedback(result={}, reward=1.0, event_id=request_id)
```
