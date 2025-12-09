# AdaptiveGraph

**Learnable Conditional Edges for Agentic Workflows**

AdaptiveGraph introduces a `LearnableEdge` primitive for agentic graph frameworks (like LangGraph). It replaces hand-written branching logic with an experience-driven, self-improving conditional edge powered by contextual bandits.

## Features

- **Self-Optimizing Routing**: Uses LinUCB to learn the best path based on context.
- **Lightweight & Fast**: No heavy RL, just efficient linear algebra logic.
- **Universal Encoder**: Handles text or arbitrary state using deterministic embedding/hashing.
- **Transparent**: Full observability into why decisions were made.

## Installation

```bash
pip install adaptivegraph
# OR
git clone ... && cd adaptivegraph && pip install -e .

# To run tests or examples with LangGraph integration:
pip install -e ".[dev]"

```

## Quick Start

```python
from adaptivegraph import LearnableEdge

# 1. Define the edge
edge = LearnableEdge(
    options=["expert_model", "fast_model"],
    policy="linucb"
)

# 2. Use it in your routing logic
# In LangGraph: graph.add_conditional_edge("start", edge)

# Simulate usage:
decision = edge("complex query about physics")
print(f"Routed to: {decision}")

# 3. Provide Feedback
# You must tell the edge how it did.
edge.record_feedback(result={}, reward=1.0) # Good choice!
```

## How It Works

1. **State Encoding**: The input (string or object) is converted to a fixed-size vector.
2. **Bandit Choice**: LinUCB estimates the potential reward for each option.
3. **Action**: The best option is returned.
4. **Learning**: When `record_feedback` is called, the model updates its internal weights to make better choices next time.

## Roadmap

- [ ] Persistent Storage (Redis/SQL)
- [ ] FAISS-based Context Memory
- [ ] Epsilon-Greedy Policy
```
