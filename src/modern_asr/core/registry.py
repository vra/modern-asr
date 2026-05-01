"""Model registry for dynamic model discovery and instantiation."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING, Callable, TypeVar

from modern_asr.core.base import ASRModel


from modern_asr.utils.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from modern_asr.core.config import BackendConfig, ModelConfig

T = TypeVar("T", bound=type[ASRModel])

# Global registry: model_id -> model class
_REGISTRY: dict[str, type[ASRModel]] = {}


def register_model(model_id: str) -> Callable[[T], T]:
    """Decorator to register an ASR model class.

    Args:
        model_id: Canonical identifier for the model (e.g. ``"fireredasr-llm"``).

    Example:
        @register_model("fireredasr-llm")
        class FireRedASRLLM(ASRModel):
            ...
    """

    def decorator(cls: T) -> T:
        if not issubclass(cls, ASRModel):
            raise TypeError(f"Class {cls.__name__} must inherit from ASRModel")
        _REGISTRY[model_id] = cls
        cls.MODEL_CARD = getattr(cls, "MODEL_CARD", "") or model_id
        return cls

    return decorator


def list_models(
    language: str | None = None,
    mode: str | None = None,
    loaded_only: bool = False,
) -> list[dict]:
    """List registered models with optional filtering.

    Args:
        language: Filter by supported language code (e.g. ``"zh"``).
        mode: Filter by supported mode (e.g. ``"transcribe"``, ``"diarize"``).
        loaded_only: If True, only return currently loaded models.

    Returns:
        List of model metadata dictionaries.
    """
    results = []
    for mid, cls in _REGISTRY.items():
        if language and language not in getattr(cls, "SUPPORTED_LANGUAGES", set()):
            continue
        if mode and mode not in getattr(cls, "SUPPORTED_MODES", set()):
            continue
        results.append(
            {
                "model_id": mid,
                "class": cls.__name__,
                "module": cls.__module__,
                "supported_languages": list(getattr(cls, "SUPPORTED_LANGUAGES", set())),
                "supported_modes": list(getattr(cls, "SUPPORTED_MODES", set())),
            }
        )
    return results


def get_model_class(model_id: str) -> type[ASRModel]:
    """Retrieve a registered model class by ID."""
    if model_id not in _REGISTRY:
        raise KeyError(
            f"Model '{model_id}' is not registered. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[model_id]


def create_model(
    model_id: str,
    config: ModelConfig,
    backend: BackendConfig | None = None,
) -> ASRModel:
    """Instantiate a model by ID.

    Args:
        model_id: Registered model identifier.
        config: Model configuration.
        backend: Optional backend override.

    Returns:
        An instantiated `ASRModel`.
    """
    cls = get_model_class(model_id)
    return cls(config=config, backend=backend)


def auto_discover_models(package_name: str = "modern_asr.models") -> None:
    """Auto-import all submodules in ``modern_asr.models`` to trigger decorators.

    This should be called once at package initialization so that model classes
    are registered without explicit imports by the user.
    """
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return

    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__, prefix=package_name + "."):
        try:
            importlib.import_module(modname)
        except Exception:
            # Skip models with unmet optional dependencies
            pass
