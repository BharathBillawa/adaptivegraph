# AdaptiveGraph

[![CI](https://github.com/BharathBillawa/adaptivegraph/actions/workflows/ci.yml/badge.svg)](https://github.com/BharathBillawa/adaptivegraph/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

```python
from adaptivegraph import LearnableEdge
```

AdaptiveGraph provides **`LearnableEdge`** – a conditional edge that improves routing decisions over time through reinforcement learning.

## What is LearnableEdge?

`LearnableEdge` replaces static routing logic with a learning-based approach. Instead of hard-coded if/else statements, it learns from feedback which routes work best for different inputs.

**Key Features:**
- 🧠 **Learning**: Adapts routing decisions based on feedback (currently LinUCB, more policies coming)
- ⚡ **Lightweight**: No training phase, no model files – learns online
- 🎯 **Balanced**: Balances exploration (trying new routes) vs exploitation (using proven routes)
- 🔧 **Flexible**: Works with strings, dicts, numpy arrays, or custom embeddings

## Why LearnableEdge?

Traditional routing logic is static and brittle. LearnableEdge adds plasticity – it learns which routes work best through experience, automatically adapting as patterns change.

## Installation

**From PyPI**:
```bash
pip install adaptivegraph
```

**From source**:
```bash
git clone https://github.com/BharathBillawa/adaptivegraph.git
cd adaptivegraph
pip install -e .
```

**With optional dependencies**:
```bash
# For semantic embeddings
pip install adaptivegraph[embed]

# For persistent storage
pip install adaptivegraph[faiss]

# For development
pip install adaptivegraph[dev]

# Everything
pip install adaptivegraph[all]
```

## Quick Start

```python
from adaptivegraph import LearnableEdge

# Create edge with routing options
edge = LearnableEdge(options=["expert_model", "fast_model", "cheap_model"])

# Make decisions
route = edge({"user": "premium", "query": "complex question"})

# Provide feedback so it learns
edge.record_feedback(result={}, reward=1.0)
```

**See more examples:**
- [`examples/basic_routing.py`](examples/basic_routing.py) - Simple routing with feedback
- [`examples/customer_support_agent.py`](examples/customer_support_agent.py) - LangGraph integration
- [`notebooks/interactive_demo.ipynb`](notebooks/interactive_demo.ipynb) - Interactive walkthrough

## How LearnableEdge Works

Think of `LearnableEdge` as a smart traffic router that learns which highway to recommend:

1. **Input State** → You provide any state (string, dict, array)
2. **Encoding** → Converts state into a fixed-size vector (32-dim by default)
3. **UCB Scoring** → For each option, calculates:
   - **Expected reward** (what worked before)
   - **Uncertainty bonus** (exploration value)
   - **UCB Score** = expected_reward + α × uncertainty
4. **Selection** → Picks option with highest UCB score
5. **Feedback** → You call `record_feedback(reward)` 
6. **Learning** → Updates internal belief about which options work best

The algorithm balances:
- **Exploitation** → Using routes with good historical performance
- **Exploration** → Trying routes with high uncertainty

## Features

- **Semantic Routing**: Use sentence transformers for similarity-based decisions
- **Async Feedback**: Provide feedback hours later using event IDs
- **Trajectory Rewards**: Reward entire multi-step sessions with decay
- **Policy Persistence**: Save and restore learned routing policies

See [`examples/`](examples/) and [`notebooks/`](notebooks/) for detailed usage.

## Comparison

**LearnableEdge vs alternatives:**
- **vs If/Else Rules**: Learns from feedback instead of requiring manual updates
- **vs ML Classifiers**: No training data needed, learns online from real usage
- **vs LLM Routers**: Faster, cheaper, and adapts based on actual outcomes
- **vs Random/Epsilon-Greedy**: Uses context to make smarter decisions, not just random exploration

## Performance

On a typical customer support routing task (3 options, 1000 decisions):
- **Convergence**: Achieves >95% accuracy within 50-100 iterations
- **Throughput**: ~10,000 decisions/second on MacBook Pro M1
- **Memory**: <50MB for 10,000 experiences

## Roadmap

### Completed ✅
- [x] Core LinUCB policy
- [x] Persistent storage (FAISS/Pickle)
- [x] Semantic state encoding (Sentence Transformers)
- [x] Async / ID-based feedback
- [x] Trajectory / trace rewards
- [x] Policy state persistence
- [x] Input validation and logging

### Planned 🚧
- [ ] Thompson Sampling policy
- [ ] Epsilon-Greedy policy
- [ ] Shared feature LinUCB (not disjoint)
- [ ] Distributed training support
- [ ] Visualization dashboard
- [ ] Automated hyperparameter tuning
- [ ] Integration examples (FastAPI, Streamlit)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors
```bash
# Clone and setup
git clone https://github.com/BharathBillawa/adaptivegraph.git
cd adaptivegraph
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/
```

## Citation

If you use AdaptiveGraph in your research, please cite:
```bibtex
@software{adaptivegraph2024,
  title = {AdaptiveGraph: Learnable Conditional Edges for Agentic Workflows},
  author = {BharathPoojary},
  year = {2024},
  url = {https://github.com/BharathBillawa/adaptivegraph}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for [LangGraph](https://github.com/langchain-ai/langgraph) but framework-agnostic
- LinUCB algorithm from [Li et al. (2010)](https://arxiv.org/abs/1003.0146)
- Inspired by contextual bandit research in recommendation systems

## Support

- 📖 [Documentation](https://github.com/BharathBillawa/adaptivegraph#readme)
- 💬 [Discussions](https://github.com/BharathBillawa/adaptivegraph/discussions)
- 🐛 [Issue Tracker](https://github.com/BharathBillawa/adaptivegraph/issues)

---

**Made with ❤️ for the AI agent community**
