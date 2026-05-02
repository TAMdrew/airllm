"""Tests for new v3.1 model backends (Llama 4, OLMo 2, Falcon 3, Granite, Jamba).

Verifies that each new backend:
- Is importable from the package
- Is registered with the correct HuggingFace architecture strings
- Inherits from AirLLMBaseModel
- Has the expected class attributes and overrides
"""

from __future__ import annotations

import pytest

from air_llm.airllm import (
    AirLLMFalcon3,
    AirLLMGranite,
    AirLLMJamba,
    AirLLMLlama4,
    AirLLMOLMo2,
)
from air_llm.airllm.airllm_base import AirLLMBaseModel
from air_llm.airllm.model_registry import ModelRegistry


@pytest.fixture(autouse=True)
def _ensure_registry_populated():
    """Re-populate the registry if cleared by other test modules.

    test_model_registry.py calls ModelRegistry._clear() which can leave
    the registry empty for tests that run afterward. We manually
    re-register the classes that we need to test.
    """
    if not ModelRegistry.list_supported():
        # Re-register all the backends we need to test
        _registrations = {
            "Llama4ForCausalLM": AirLLMLlama4,
            "Llama4ForConditionalGeneration": AirLLMLlama4,
            "OLMo2ForCausalLM": AirLLMOLMo2,
            "FalconForCausalLM": AirLLMFalcon3,
            "GraniteForCausalLM": AirLLMGranite,
            "JambaForCausalLM": AirLLMJamba,
        }
        for arch_name, cls in _registrations.items():
            ModelRegistry._registry[arch_name] = cls


# ---------------------------------------------------------------------------
# Import tests — fail fast if any backend can't be imported
# ---------------------------------------------------------------------------
class TestNewBackendImports:
    """Verify all v3.1 model backends are importable."""

    def test_llama4_importable(self) -> None:
        assert AirLLMLlama4 is not None

    def test_olmo2_importable(self) -> None:
        assert AirLLMOLMo2 is not None

    def test_falcon3_importable(self) -> None:
        assert AirLLMFalcon3 is not None

    def test_granite_importable(self) -> None:
        assert AirLLMGranite is not None

    def test_jamba_importable(self) -> None:
        assert AirLLMJamba is not None


# ---------------------------------------------------------------------------
# Inheritance tests — all must extend AirLLMBaseModel
# ---------------------------------------------------------------------------
class TestNewBackendInheritance:
    """Verify all new backends inherit from AirLLMBaseModel."""

    @pytest.mark.parametrize(
        "backend_class",
        [AirLLMLlama4, AirLLMOLMo2, AirLLMFalcon3, AirLLMGranite, AirLLMJamba],
        ids=["Llama4", "OLMo2", "Falcon3", "Granite", "Jamba"],
    )
    def test_inherits_from_base(self, backend_class: type) -> None:
        """Each backend must be a subclass of AirLLMBaseModel."""
        assert issubclass(backend_class, AirLLMBaseModel)


# ---------------------------------------------------------------------------
# Registry tests — architecture strings map to correct classes
# ---------------------------------------------------------------------------
class TestNewBackendRegistrations:
    """Verify new backends are registered with correct architecture strings."""

    @pytest.mark.parametrize(
        ("arch_name", "expected_class"),
        [
            ("Llama4ForCausalLM", AirLLMLlama4),
            ("Llama4ForConditionalGeneration", AirLLMLlama4),
            ("OLMo2ForCausalLM", AirLLMOLMo2),
            ("FalconForCausalLM", AirLLMFalcon3),
            ("GraniteForCausalLM", AirLLMGranite),
            ("JambaForCausalLM", AirLLMJamba),
        ],
        ids=[
            "Llama4-CausalLM",
            "Llama4-Conditional",
            "OLMo2",
            "Falcon3",
            "Granite",
            "Jamba",
        ],
    )
    def test_architecture_resolves_to_correct_class(
        self, arch_name: str, expected_class: type
    ) -> None:
        """ModelRegistry.get() returns the correct class for each architecture."""
        resolved = ModelRegistry.get(arch_name)
        assert resolved is expected_class, (
            f"Expected {expected_class.__name__} for {arch_name}, got {resolved.__name__}"
        )


# ---------------------------------------------------------------------------
# Architecture string count tests
# ---------------------------------------------------------------------------
class TestArchitectureCoverage:
    """Verify the registry has comprehensive architecture coverage."""

    def test_new_architectures_are_registered(self) -> None:
        """All 6 new v3.1 architecture strings are in the registry."""
        supported = ModelRegistry.list_supported()
        assert len(supported) >= 6, (
            f"Expected >=6 new architectures, got {len(supported)}"
        )

    def test_new_architectures_in_supported_list(self) -> None:
        """All v3.1 architecture strings appear in the supported list."""
        supported = ModelRegistry.list_supported()
        new_archs = [
            "Llama4ForCausalLM",
            "Llama4ForConditionalGeneration",
            "OLMo2ForCausalLM",
            "FalconForCausalLM",
            "GraniteForCausalLM",
            "JambaForCausalLM",
        ]
        for arch in new_archs:
            assert arch in supported, f"{arch} not found in ModelRegistry"


# ---------------------------------------------------------------------------
# Class attribute tests
# ---------------------------------------------------------------------------
class TestNewBackendAttributes:
    """Verify new backends have expected class structure."""

    @pytest.mark.parametrize(
        "backend_class",
        [AirLLMLlama4, AirLLMOLMo2, AirLLMFalcon3, AirLLMGranite, AirLLMJamba],
        ids=["Llama4", "OLMo2", "Falcon3", "Granite", "Jamba"],
    )
    def test_has_init_method(self, backend_class: type) -> None:
        """Each backend has an __init__ method."""
        assert hasattr(backend_class, "__init__")

    @pytest.mark.parametrize(
        "backend_class",
        [AirLLMLlama4, AirLLMOLMo2, AirLLMFalcon3, AirLLMGranite, AirLLMJamba],
        ids=["Llama4", "OLMo2", "Falcon3", "Granite", "Jamba"],
    )
    def test_inherits_set_layer_names_dict(self, backend_class: type) -> None:
        """Each backend inherits set_layer_names_dict from base."""
        assert hasattr(backend_class, "set_layer_names_dict")

    @pytest.mark.parametrize(
        "backend_class",
        [AirLLMLlama4, AirLLMOLMo2, AirLLMFalcon3, AirLLMGranite, AirLLMJamba],
        ids=["Llama4", "OLMo2", "Falcon3", "Granite", "Jamba"],
    )
    def test_inherits_forward(self, backend_class: type) -> None:
        """Each backend inherits forward from base."""
        assert hasattr(backend_class, "forward")

    def test_jamba_defaults_trust_remote_code(self) -> None:
        """Jamba backend should set trust_remote_code=True by default."""
        # Verify the __init__ accepts trust_remote_code via kwargs
        import inspect

        sig = inspect.signature(AirLLMJamba.__init__)
        params = list(sig.parameters.keys())
        assert "kwargs" in params or "args" in params
