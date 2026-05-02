"""Decorator-based model registry for mapping HuggingFace architectures to AirLLM classes.

This module replaces the long if/elif chain in ``auto_model.py`` with a
dynamic registry pattern.  Each model class registers itself via the
:pyfunc:`ModelRegistry.register` decorator, and ``AutoModel`` resolves the
correct class at runtime through :pyfunc:`ModelRegistry.get`.

Example
-------
::

    @ModelRegistry.register("LlamaForCausalLM", "LLaMAForCausalLM")
    class AirLLMLlama2(AirLLMBaseModel):
        ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=type)


class ModelRegistry:
    """Central registry mapping HuggingFace architecture names to model classes.

    The registry is populated at import-time via the ``@ModelRegistry.register``
    decorator applied to each concrete model class.
    """

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, *architecture_names: str) -> Callable[[_T], _T]:
        """Register a model class for one or more HuggingFace architecture names.

        Args:
            *architecture_names: One or more architecture strings as they appear
                in a HuggingFace ``config.json`` ``architectures`` list
                (e.g. ``"LlamaForCausalLM"``).

        Returns:
            A class decorator that registers *model_class* and returns it
            unchanged.

        Raises:
            ValueError: If an architecture name is already registered to a
                *different* class.
        """

        def decorator(model_class: _T) -> _T:
            for name in architecture_names:
                existing = cls._registry.get(name)
                if existing is not None and existing is not model_class:
                    raise ValueError(
                        f"Architecture {name!r} is already registered to "
                        f"{existing.__name__}, cannot re-register to "
                        f"{model_class.__name__}"  # type: ignore[union-attr]
                    )
                cls._registry[name] = model_class  # type: ignore[assignment]
                logger.debug("Registered architecture %r -> %s", name, model_class.__name__)
            return model_class

        return decorator

    @classmethod
    def get(cls, architecture_name: str) -> type:
        """Look up the model class for *architecture_name*.

        Unlike the old if/elif chain this performs an *exact* match first, then
        falls back to a substring search for backward compatibility (e.g.
        ``"QWen"`` matching ``"QWenLMHeadModel"``).

        Args:
            architecture_name: The architecture string from HuggingFace config.

        Returns:
            The registered model class.

        Raises:
            ValueError: If no registered architecture matches.
        """
        # Exact match first.
        if architecture_name in cls._registry:
            return cls._registry[architecture_name]

        # Substring fallback (preserves old auto_model.py behaviour).
        for registered_name, model_class in cls._registry.items():
            if registered_name in architecture_name:
                logger.debug(
                    "Substring match: %r matched via registered key %r",
                    architecture_name,
                    registered_name,
                )
                return model_class

        supported = cls.list_supported()
        raise ValueError(f"Unsupported architecture: {architecture_name!r}. Supported: {supported}")

    @classmethod
    def list_supported(cls) -> list[str]:
        """Return a sorted list of all registered architecture names."""
        return sorted(cls._registry.keys())

    @classmethod
    def _clear(cls) -> None:
        """Reset the registry.  **Test-only** — not part of public API."""
        cls._registry.clear()
