# Copilot Instructions for AdaptiveGraph

Purpose: Equip AI coding agents to be immediately productive in this repo by capturing the architecture, workflows, and project-specific patterns.

## Big Picture
- **Goal:** Learnable conditional edge for agentic workflows; replaces hand-coded branching with a contextual bandit.
- **Core primitive:** `LearnableEdge` (see `src/adaptivegraph/core.py`) that:
  - Encodes state via `StateEncoder` (`encoder.py`).
  - Selects an option using LinUCB bandit policy (`policy.py`).
  - Records feedback in an experience store (`memory.py`) and updates the policy.
- **Flow:** input → encode → select_action → return option → record_feedback(result, reward) → update policy + memory.
- **Design trade-offs:** Lightweight, deterministic encoding by default to avoid heavy dependencies; disjoint LinUCB per arm for transparency and speed.

## Key Files
- `src/adaptivegraph/core.py`: `LearnableEdge` orchestration and feedback lifecycle.
- `src/adaptivegraph/encoder.py`: `StateEncoder` with pass-through vectors or deterministic hashed embeddings for strings/objects.
- `src/adaptivegraph/policy.py`: `LinUCBPolicy` (disjoint arms) with ridge init (I) and `alpha` exploration.
- `src/adaptivegraph/memory.py`: Simple `InMemoryExperienceStore` for contexts/actions/rewards.
- `examples/basic_routing.py`: End-to-end demo of usage and reward loop.
- `debug_simulation.py`: Step-by-step UCB scoring introspection for two routes.
- `tests/test_convergence.py`: Convergence test that validates learning on two contexts.

## Usage Pattern
- Instantiate:
  ```python
  from adaptivegraph import LearnableEdge
  edge = LearnableEdge(options=["slow_accurate", "fast_cheap"], policy="linucb", exploration_alpha=0.5)
  ```
- Decision:
  ```python
  route = edge({"user": "vip", "query": "..."})
  ```
- Feedback (critical):
  ```python
  reward = 1.0 if route == "slow_accurate" else 0.0
  edge.record_feedback(result={}, reward=reward)
  ```
- Optional: use `value_key` to encode a specific field from dict-like state.

## Developer Workflows
- **Python version:** `>=3.11` (see `pyproject.toml`).
- **Install:**
  ```bash
  pip install -e .
  # optional dev extras
  pip install -e .[dev]
  ```
- **Run example:**
  ```bash
  python examples/basic_routing.py
  ```
- **Debug introspection:**
  ```bash
  python debug_simulation.py
  ```
- **Run tests:**
  ```bash
  python -m unittest tests/test_convergence.py -v
  # or discover
  python -m unittest discover -s tests -p "test_*.py" -v
  ```

## Conventions & Patterns
- **Encoding:**
  - Strings/dicts: deterministic hashed vector via `sha256`-seeded RNG; normalized.
  - Numpy arrays: pass-through then truncate/flatten to `feature_dim`.
  - Custom embeddings: supply `embedding_fn` returning array-like.
- **Policy:** `LinUCBPolicy` with per-arm `(A, b)`; selection uses `theta = A^-1 b` and UCB term `alpha * sqrt(x^T A^-1 x)`.
- **Feedback lifecycle:** `__call__` caches last context+action on the instance; `record_feedback` consumes and clears them. In async/multi-threaded environments, wrap per-request or add request IDs.
- **Options:** Keep `options` order stable; actions index maps directly to names by position.

## Integration Notes
- **LangGraph-style conditional edges:** Use `LearnableEdge` where `add_conditional_edge` expects a callable returning next node label.
- **State objects:** If routing state is an object/dict, prefer setting `value_key` to the field carrying semantic signal (e.g., `"user_type"`).
- **Observability:** Internals (`policy.A`, `policy.b`) are accessible for diagnostics, as shown in `debug_simulation.py`.

## Common Gotchas
- Forgetting to call `record_feedback`: model won’t learn; tests/examples always include it.
- Passing very high-dimensional arrays: encoder truncates; ensure `feature_dim` matches expectations.
- Multi-request concurrency: current implementation is not thread-safe; add request scoping before using in servers.

## Extending
- To add new policies, implement the `BanditPolicy` protocol (methods: `select_action`, `update`) and wire selection in `LearnableEdge.__init__`.
- To persist experience, implement `ExperienceStore` and replace `InMemoryExperienceStore`.

---
Feedback: If anything here is unclear or missing, tell me what you need (workflows, integration examples, or policy/encoder extension guidance) and I’ll refine this doc.
