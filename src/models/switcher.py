"""
Model switching utilities for runtime model changes.

Provides functions to hot-swap models without restarting the application,
with proper cleanup and memory management.
"""

import logging
from typing import Any
from .loader import ModelInfo, ModelLoader
from .exceptions import ModelLoadError, ModelNotLoadedError
from .memory import MemoryManager

logger = logging.getLogger(__name__)


def switch_model(
    loader: ModelLoader,
    new_model_name: str,
    quantization: str = "4bit",
    warmup: bool = True,
) -> tuple[bool, str, ModelInfo | None]:
    """
    Switch to a different model at runtime.

    This function handles the complete model switching workflow:
    1. Validates the new model exists in registry
    2. Checks memory availability
    3. Unloads current model (if any)
    4. Loads new model
    5. Performs warmup (optional)

    Args:
        loader: ModelLoader instance managing the current model
        new_model_name: Name of the model to switch to
        quantization: Quantization level for new model (default: "4bit")
        warmup: Whether to warmup new model (default: True)

    Returns:
        Tuple of (success, message, model_info)
        - success: True if switch succeeded
        - message: Status or error message
        - model_info: ModelInfo for new model if successful, None otherwise

    Example:
        >>> from src.models import ModelLoader, switch_model
        >>> from src.config import ModelConfig
        >>> loader = ModelLoader(ModelConfig(model_name="phi-3-mini"))
        >>> loader.load()
        >>> success, msg, info = switch_model(loader, "mistral-7b")
        >>> print(msg)
        "✓ Switched to mistral-7b (4bit) on mps - 4500MB"
    """
    from .registry import get_model_spec, validate_model_name
    from ..devices import DeviceDetector

    logger.info(f"Attempting to switch model to: {new_model_name}")

    try:
        # 1. Validate new model name
        is_valid, validation_msg = validate_model_name(new_model_name)
        if not is_valid:
            logger.error(f"Invalid model name: {validation_msg}")
            return (False, f"✗ {validation_msg}", None)

        # 2. Get new model spec
        spec = get_model_spec(new_model_name)
        if spec is None:
            return (False, f"✗ Model '{new_model_name}' not found in registry", None)

        # 3. Check memory availability
        can_fit, mem_message = MemoryManager.can_fit_model(
            new_model_name, quantization, "mps"
        )

        if not can_fit:
            logger.warning(f"Memory check: {mem_message}")
            return (False, f"✗ Memory check failed: {mem_message}", None)

        logger.info(mem_message)

        # 4. Get current model name (if any)
        current_model = None
        if loader.is_loaded():
            try:
                current_info = loader.get_info()
                current_model = current_info.model_name
                logger.info(f"Unloading current model: {current_model}")
            except Exception as e:
                logger.warning(f"Could not get current model info: {e}")

        # 5. Unload current model
        if loader.is_loaded():
            loader.unload()
            logger.info("Current model unloaded")

        # 6. Create new config
        # We need to update the loader's config attributes
        loader.model_name = new_model_name
        loader.quantization = quantization
        loader.warmup = warmup
        loader.device = "auto"  # Auto-detect device
        loader.model_path_override = None  # Use registry path

        # 7. Load new model
        logger.info(f"Loading new model: {new_model_name}")
        model_info = loader.load()

        # 8. Success message
        success_msg = (
            f"✓ Switched from '{current_model}' to '{model_info.model_name}' "
            f"({model_info.quantization}) on {model_info.device} - "
            f"{model_info.memory_usage_mb}MB"
            if current_model
            else f"✓ Loaded '{model_info.model_name}' ({model_info.quantization}) "
            f"on {model_info.device} - {model_info.memory_usage_mb}MB"
        )

        logger.info(success_msg)
        return (True, success_msg, model_info)

    except ModelLoadError as e:
        error_msg = f"✗ Failed to load new model: {str(e)}"
        logger.error(error_msg)
        return (False, error_msg, None)

    except Exception as e:
        error_msg = f"✗ Unexpected error during model switch: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (False, error_msg, None)


def validate_switch_feasibility(
    current_model_name: str,
    new_model_name: str,
    quantization: str = "4bit",
) -> tuple[bool, str]:
    """
    Check if switching to a new model is feasible without actually switching.

    Validates that:
    1. New model exists in registry
    2. New model is different from current model
    3. Sufficient memory is available

    Args:
        current_model_name: Name of currently loaded model
        new_model_name: Name of model to switch to
        quantization: Quantization level for new model

    Returns:
        Tuple of (is_feasible, message)
        - is_feasible: True if switch should succeed
        - message: Explanation or warning

    Example:
        >>> is_feasible, msg = validate_switch_feasibility(
        ...     "phi-3-mini", "mistral-7b", "4bit"
        ... )
        >>> print(msg)
        "✓ Switch feasible: mistral-7b requires ~4500MB, 8192MB available"
    """
    from .registry import get_model_spec, validate_model_name

    # 1. Check if models are different
    if current_model_name == new_model_name:
        return (False, "✗ New model is the same as current model")

    # 2. Validate new model
    is_valid, validation_msg = validate_model_name(new_model_name)
    if not is_valid:
        return (False, f"✗ {validation_msg}")

    # 3. Get model spec
    spec = get_model_spec(new_model_name)
    if spec is None:
        return (False, f"✗ Model '{new_model_name}' not found in registry")

    # 4. Check memory
    can_fit, mem_message = MemoryManager.can_fit_model(
        new_model_name, quantization, "mps"
    )

    if can_fit:
        return (True, f"✓ Switch feasible: {mem_message}")
    else:
        return (False, f"✗ Switch not feasible: {mem_message}")


def get_switch_recommendation(
    current_memory_mb: int,
    target_use_case: str = "general",
) -> tuple[str, str]:
    """
    Recommend a model to switch to based on current memory usage and use case.

    Args:
        current_memory_mb: Current model memory usage in MB
        target_use_case: Desired use case (chat, long_context, general, etc.)

    Returns:
        Tuple of (recommended_model, reason)
        - recommended_model: Model name to switch to
        - reason: Why this model is recommended

    Example:
        >>> model, reason = get_switch_recommendation(2500, "long_context")
        >>> print(f"{model}: {reason}")
        "phi-3-mini-128k: 128K context window, similar memory footprint"
    """
    from .registry import RECOMMENDED_MODELS, get_model_spec

    # Get base recommendation for use case
    recommended = RECOMMENDED_MODELS.get(target_use_case, "phi-3-mini")
    spec = get_model_spec(recommended)

    if spec is None:
        return ("phi-3-mini", "Default model")

    # Check if recommended model fits in similar memory budget
    recommended_memory = spec["memory_estimate_mb"]

    # If significantly more memory needed, suggest lighter alternative
    if recommended_memory > current_memory_mb * 1.5:
        # Find lighter model for use case
        lighter_alternatives = {
            "long_context": "phi-3-mini-128k",  # Still has long context
            "general": "phi-3-mini",  # Lighter general model
            "quality": "mistral-7b",  # Lighter than llama but good quality
            "multilingual": "qwen-2.5-3b",  # Lighter multilingual
        }

        alternative = lighter_alternatives.get(target_use_case, "phi-3-mini")
        alt_spec = get_model_spec(alternative)

        if alt_spec:
            return (
                alternative,
                f"Optimized for {target_use_case} with similar memory footprint "
                f"(~{alt_spec['memory_estimate_mb']}MB vs current {current_memory_mb}MB)",
            )

    # Return recommended model
    reason = f"Best for {target_use_case}"
    if spec.get("description"):
        reason = spec["description"]

    return (recommended, reason)


def cleanup_before_switch() -> None:
    """
    Perform cleanup operations before switching models.

    This function:
    1. Clears Metal memory cache
    2. Resets peak memory tracking
    3. Forces garbage collection (if needed)

    Should be called before switching to ensure maximum available memory.

    Example:
        >>> from src.models import cleanup_before_switch
        >>> cleanup_before_switch()
        # Metal cache cleared, ready for model switch
    """
    logger.info("Performing cleanup before model switch...")

    # Clear Metal cache
    MemoryManager.clear_cache()

    # Reset peak memory tracking
    MemoryManager.reset_peak_memory()

    # Optional: force Python garbage collection
    try:
        import gc

        gc.collect()
        logger.debug("Garbage collection performed")
    except Exception as e:
        logger.debug(f"Could not force garbage collection: {e}")

    logger.info("Cleanup complete")
