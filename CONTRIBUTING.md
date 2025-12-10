# Contributing to AdaptiveGraph

Thank you for your interest in contributing to AdaptiveGraph! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites
- Python 3.11 or higher
- Git

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/adaptivegraph.git
   cd adaptivegraph
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode with all extras:**
   ```bash
   pip install -e ".[dev,embed,faiss,all]"
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Running Tests

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Run specific test files:
```bash
python -m pytest tests/test_convergence.py -v
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=adaptivegraph --cov-report=html
```

### Code Style

We follow PEP 8 style guidelines. Key points:
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for function signatures
- Write docstrings for all public APIs

Format your code:
```bash
black src/ tests/
isort src/ tests/
```

Lint your code:
```bash
flake8 src/ tests/
mypy src/
```

### Running Examples

Test your changes with examples:
```bash
python examples/basic_routing.py
python examples/customer_support_agent.py
python debug_simulation.py
```

## Contribution Guidelines

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Provide a clear, descriptive title
3. Include:
   - Python version
   - AdaptiveGraph version
   - Minimal reproducible example
   - Expected vs actual behavior
   - Full error traceback (if applicable)

### Submitting Pull Requests

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add awesome new feature"
   ```

   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test additions/changes
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

4. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **In your PR description, include:**
   - What changes were made and why
   - How to test the changes
   - Any related issues (use "Fixes #123")

## Areas for Contribution

### High Priority
- [ ] Additional bandit policies (Thompson Sampling, Epsilon-Greedy)
- [ ] Distributed training support
- [ ] Performance optimizations
- [ ] More comprehensive documentation

### Good First Issues
- [ ] Additional test cases
- [ ] Example improvements
- [ ] Documentation typos/clarity
- [ ] Error message improvements

### Advanced Features
- [ ] Shared feature LinUCB (not disjoint)
- [ ] Automated hyperparameter tuning
- [ ] Integration with popular frameworks (FastAPI, Streamlit)
- [ ] Visualization dashboard

## Code Review Process

All submissions require review. We look for:
- Code quality and style
- Test coverage
- Documentation completeness
- Backward compatibility
- Performance implications

## Testing Requirements

All PRs must:
- Pass existing tests
- Include new tests for new functionality
- Maintain >80% code coverage
- Pass linting checks

## Documentation Standards

- All public classes/methods need docstrings
- Use Google-style docstring format
- Include type hints
- Provide usage examples for complex features
- Update README.md if adding major features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open a discussion on GitHub
- Ask in pull request comments
- Reach out to maintainers

Thank you for contributing to AdaptiveGraph! 🚀
