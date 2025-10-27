"""
Model management for MLX on Apple Silicon.

Provides model loading, memory management, and device detection
for running quantized language models efficiently on Apple Silicon.
"""

from .loader import ModelLoader, ModelInfo
from .exceptions import (
    ModelError,
    ModelLoadError,
    OutOfMemoryError,
    DeviceError,
    ModelNotFoundError,
    ModelNotLoadedError,
    QuantizationError,
)
from .memory import MemoryManager
from .registry import (
    MLX_SUPPORTED_MODELS,
    get_model_spec,
    list_available_models,
    get_models_by_size,
    format_model_list,
    validate_model_name,
    get_recommended_model,
    DEFAULT_MODEL,
    RECOMMENDED_MODELS,
)
# Wave 2: Health checks, model switching, and tools
from .health import (
    check_model_loaded,
    check_metal_backend,
    get_health_status,
    format_health_report,
    HealthStatus,
)
from .switcher import (
    switch_model,
    validate_switch_feasibility,
    get_switch_recommendation,
    cleanup_before_switch,
)
from .tools import (
    list_models,
    get_model_info,
    download_model,
    compare_models,
    find_models_by_description,
    get_model_recommendation,
    format_model_info,
    check_model_compatibility,
    ModelMetadata,
)

__all__ = [
    # Core loader
    "ModelLoader",
    "ModelInfo",
    # Exceptions
    "ModelError",
    "ModelLoadError",
    "OutOfMemoryError",
    "DeviceError",
    "ModelNotFoundError",
    "ModelNotLoadedError",
    "QuantizationError",
    # Memory management
    "MemoryManager",
    # Registry
    "MLX_SUPPORTED_MODELS",
    "get_model_spec",
    "list_available_models",
    "get_models_by_size",
    "format_model_list",
    "validate_model_name",
    "get_recommended_model",
    "DEFAULT_MODEL",
    "RECOMMENDED_MODELS",
    # Wave 2: Health checks
    "check_model_loaded",
    "check_metal_backend",
    "get_health_status",
    "format_health_report",
    "HealthStatus",
    # Wave 2: Model switching
    "switch_model",
    "validate_switch_feasibility",
    "get_switch_recommendation",
    "cleanup_before_switch",
    # Wave 2: Model tools
    "list_models",
    "get_model_info",
    "download_model",
    "compare_models",
    "find_models_by_description",
    "get_model_recommendation",
    "format_model_info",
    "check_model_compatibility",
    "ModelMetadata",
]
