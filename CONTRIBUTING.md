# Contributing to AirLLM

Thank you for your interest in contributing to AirLLM!
This guide covers everything you need to get started.

> **Core Philosophy:** AirLLM enables 100% local, free LLM inference. No API keys required for inference, no subscriptions, no cloud dependencies. When contributing, ensure all new features maintain this local-first approach. HuggingFace libraries run entirely locally for config parsing, tokenization, and one-time weight download. After initial download, everything runs offline. Zero subscription fees — your hardware, your models, forever.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Model Backend](#adding-a-new-model-backend)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git
- A CUDA-capable GPU (for integration testing) or Apple Silicon Mac

### Clone and Install

```bash
git clone https://github.com/lyogavin/airllm.git
cd airllm
pip install -e ".[dev]"
```

This installs AirLLM in editable mode with all development dependencies:

- `pytest` — test runner
- `pytest-cov` — coverage reporting
- `ruff` — linter and formatter
- `mypy` — static type checking

### Verify Installation

```bash
# Run linter
ruff check air_llm/airllm/

# Run type checker
mypy air_llm/airllm/

# Run tests
pytest
```

## Project Structure

```
airllm_v2/
├── air_llm/
│   ├── airllm/
│   │   ├── __init__.py            # Package exports, version
│   │   ├── auto_model.py          # AutoModel factory
│   │   ├── model_registry.py      # @ModelRegistry.register decorator
│   │   ├── airllm_base.py         # AirLLMBaseModel (core inference loop)
│   │   ├── constants.py           # Centralized constants
│   │   ├── utils.py               # Utilities (split layers, memory)
│   │   ├── profiler.py            # LayeredProfiler
│   │   ├── airllm.py              # Llama backend
│   │   ├── airllm_gemma4.py       # Gemma 4 backend
│   │   ├── airllm_qwen3.py        # Qwen 3 backend
│   │   ├── airllm_deepseek.py     # DeepSeek V2/V3 backend
│   │   ├── ...                    # Other model backends
│   │   └── persist/               # Model persistence layer
│   └── tests/                     # Test suite
├── docs/                          # Architecture and reference docs
├── pyproject.toml                 # Packaging and tool configuration
├── README.md                      # Project README
├── CONTRIBUTING.md                # This file
└── CHANGELOG.md                   # Version history
```

## Adding a New Model Backend

Most new models can be added in under 20 lines of code.
See [Model Integration Guide](docs/MODEL_INTEGRATION.md) for the full reference.

### Step 1: Check if Code is Needed

If the new model uses the standard layer naming convention (`model.embed_tokens`, `model.layers`, `model.norm`, `lm_head`), you only need a minimal class with the [`@ModelRegistry.register`](air_llm/airllm/model_registry.py:37) decorator.

### Step 2: Create the Backend File

Create `air_llm/airllm/airllm_<model>.py`:

```python
"""AirLLM backend for <ModelName> architecture."""

from __future__ import annotations

import logging
from typing import Any

from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("ModelNameForCausalLM")  # From config.json "architectures"
class AirLLMModelName(AirLLMBaseModel):
    """AirLLM implementation for <ModelName> models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self) -> bool:
        """<ModelName> is incompatible with BetterTransformer."""
        return False

    def get_generation_config(self) -> GenerationConfig:
        """Return a bare GenerationConfig for <ModelName>."""
        return GenerationConfig()
```

### Step 3: Register the Import

Add your import to [`auto_model.py`](air_llm/airllm/auto_model.py:23):

```python
from . import (
    airllm_modelname,  # noqa: F401
    # ... existing imports
)
```

And to [`__init__.py`](air_llm/airllm/__init__.py:1):

```python
from .airllm_modelname import AirLLMModelName
```

### Step 4: Handle Non-Standard Layer Names

If the model uses different layer naming (like ChatGLM or QWen v1), override [`set_layer_names_dict()`](air_llm/airllm/airllm_base.py:128):

```python
def set_layer_names_dict(self) -> None:
    self.layer_names_dict = {
        "embed": "transformer.word_embeddings",
        "layer_prefix": "transformer.layers",
        "norm": "transformer.final_layernorm",
        "lm_head": "output_layer",
    }
```

### Step 5: Add Tests

Add test cases to [`test_model_backends.py`](air_llm/tests/test_model_backends.py):

```python
def test_modelname_registration(self):
    """Verify ModelName architecture is registered."""
    cls = ModelRegistry.get("ModelNameForCausalLM")
    assert cls is AirLLMModelName
```

### Step 6: Update Documentation

- Add the model to the [Supported Models](README.md#-supported-models) table in the README.
- Add a detailed entry in [`docs/SUPPORTED_MODELS.md`](docs/SUPPORTED_MODELS.md).

## Code Style

AirLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Formatting and Linting

```bash
# Check for issues
ruff check air_llm/airllm/

# Auto-fix issues
ruff check --fix air_llm/airllm/

# Format code
ruff format air_llm/airllm/
```

### Type Checking

```bash
mypy air_llm/airllm/
```

### Key Style Rules

- **Type hints** on all public function signatures.
- **Docstrings** on all public classes and methods (Google-style).
- **Logging** via `logging.getLogger(__name__)` — never `print()`.
- **Constants** in [`constants.py`](air_llm/airllm/constants.py:1) — no magic numbers in logic.
- **Line length** of 100 characters (configured in [`pyproject.toml`](pyproject.toml:67)).
- **Imports** ordered: stdlib → third-party → local (enforced by ruff isort).

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Phi-4 model backend
fix: correct attention mask shape for ChatGLM
docs: update VRAM requirements table
refactor: extract layer loading into base class method
test: add registry collision detection test
chore: update ruff to 0.5.0
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest air_llm/tests/test_model_registry.py

# Run with coverage
pytest --cov=airllm --cov-report=term-missing

# Run only tests matching a pattern
pytest -k "test_gemma"
```

### Test Structure

Tests are organized by module:

| Test File                                                                | Coverage                                        |
| ------------------------------------------------------------------------ | ----------------------------------------------- |
| [`test_model_registry.py`](air_llm/tests/test_model_registry.py)         | Registry decorator, lookup, collision detection |
| [`test_automodel.py`](air_llm/tests/test_automodel.py)                   | AutoModel dispatching, fallback behavior        |
| [`test_constants.py`](air_llm/tests/test_constants.py)                   | Constants values, platform detection            |
| [`test_model_layer_config.py`](air_llm/tests/test_model_layer_config.py) | ModelLayerConfig dataclass                      |
| [`test_model_backends.py`](air_llm/tests/test_model_backends.py)         | All model backend registrations                 |
| [`test_compression.py`](air_llm/tests/test_compression.py)               | Quantization utilities                          |
| [`test_persist.py`](air_llm/tests/test_persist.py)                       | Model persistence layer                         |
| [`test_utils.py`](air_llm/tests/test_utils.py)                           | Utility functions                               |

### Writing Tests

Follow the **Arrange-Act-Assert** pattern:

```python
def test_registry_exact_match(self):
    """ModelRegistry.get returns the correct class for an exact match."""
    # Arrange
    expected = AirLLMLlama2

    # Act
    result = ModelRegistry.get("LlamaForCausalLM")

    # Assert
    assert result is expected
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Implement** your changes following the style guidelines above.
3. **Add tests** for any new functionality.
4. **Run the full test suite** — all tests must pass.
5. **Run linting** — `ruff check` and `mypy` must be clean.
6. **Update documentation** if your change affects the public API or supported models.
7. **Write a clear PR description** explaining:
   - What the change does.
   - Why the change is needed.
   - How it was tested.
8. **Reference any related issues** using `Closes #123` or `Fixes #456`.

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linter passes (`ruff check`)
- [ ] Type checker passes (`mypy`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional commits
- [ ] No new `print()` statements (use `logging`)
- [ ] No bare `except` blocks (use specific exceptions)

## Issue Guidelines

### Bug Reports

Include the following:

- Python version (`python --version`)
- AirLLM version (`python -c "import airllm; print(airllm.__version__)"`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- GPU info (`nvidia-smi` output or Mac model)
- Full error traceback
- Minimal reproduction script

### Feature Requests

Describe:

- The use case — what problem does this solve?
- Proposed solution — how should it work?
- Alternatives considered — what else did you evaluate?

### Model Support Requests

For new model support, include:

- Model name and HuggingFace repo ID.
- Architecture string from `config.json` (the `"architectures"` field).
- Model size(s) you need supported.
- Whether the model uses standard or custom layer naming.

---

Thank you for contributing to AirLLM! 🌬️
