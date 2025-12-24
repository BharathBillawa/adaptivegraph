# Changelog

All notable changes to AdaptiveGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-12-24

### Fixed
- Improved factory method `LearnableEdge.create` with clear error messages for missing optional dependencies (`sentence-transformers`, `faiss-cpu`)
- Implemented missing statistics methods `get_statistics`, `total_decisions`, and `state_history` in `InMemoryExperienceStore` and `FaissExperienceStore`

### Documentation
- Corrected Async/Trajectory usage examples to show `event_id`/`trace_id` inside the state dictionary
- Added Statistics API examples

### Added
- Policy state persistence with `save_policy()` and `load_policy()` methods
- Comprehensive input validation for all parameters
- Logging support throughout the codebase
- Warning for vector truncation in StateEncoder
- `ridge_lambda` parameter for LinUCB regularization control
- `auto_save` parameter for FaissExperienceStore performance optimization
- Detailed docstrings for all public APIs
- Reward validation (checks for NaN/inf values)

### Fixed (Internal)
- Removed duplicate `@classmethod` decorator in LearnableEdge
- Fixed indentation issues in factory method
- Improved error messages with suggested valid options

### Changed
- Enhanced StateEncoder with better documentation about fallback hashing limitations
- Improved LinUCBPolicy with ridge_lambda parameter
- FaissExperienceStore now supports manual save for better performance

### Security
- Added validation to prevent invalid inputs causing crashes

## [0.1.0] - 2024-12-10

### Added
- Initial release of AdaptiveGraph
- Core `LearnableEdge` class for learnable conditional routing
- LinUCB policy implementation (disjoint arms)
- StateEncoder with multiple encoding strategies
- InMemoryExperienceStore and FaissExperienceStore
- SentenceTransformer embedding support
- Async feedback via event_id
- Trajectory rewards via trace_id
- ErrorScorer and LLMScorer for automated reward calculation
- Factory method `LearnableEdge.create()` for easy setup
- Comprehensive examples (basic_routing, customer_support_agent)
- Test suite with convergence tests
- MIT License
- Documentation and README

### Features
- Sequential, async (event_id), and trajectory (trace_id) feedback modes
- Optional dependencies for sentence-transformers and faiss
- Deterministic hashing fallback for encoding
- LangGraph integration support

[Unreleased]: https://github.com/BharathBillawa/adaptivegraph/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BharathBillawa/adaptivegraph/releases/tag/v0.1.0
