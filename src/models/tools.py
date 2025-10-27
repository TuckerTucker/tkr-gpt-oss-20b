"""
Model utility tools for listing, downloading, and managing MLX models.

Provides user-facing tools for discovering available models, getting model
information, and downloading models from HuggingFace.
"""

import logging
from typing import TypedDict
from .registry import (
    MLX_SUPPORTED_MODELS,
    list_available_models,
    get_models_by_size,
    format_model_list,
    get_model_spec,
    get_recommended_model,
)
from .memory import MemoryManager

logger = logging.getLogger(__name__)


class ModelMetadata(TypedDict):
    """Extended model metadata with additional computed fields."""

    name: str
    path: str
    context_length: int
    quantization: str
    memory_estimate_mb: int
    memory_estimate_gb: float
    description: str
    context_tokens_k: int
    size_category: str


def list_models(
    filter_by_size: str | None = None,
    filter_by_context: int | None = None,
    sort_by: str = "name",
) -> list[str]:
    """
    List available MLX models with optional filtering and sorting.

    Args:
        filter_by_size: Filter by size category ("small", "medium", "large")
        filter_by_context: Filter by minimum context length (in tokens)
        sort_by: Sort criterion ("name", "memory", "context")

    Returns:
        List of model names matching filters

    Example:
        >>> models = list_models(filter_by_size="small")
        >>> print(models)
        ['gemma-2-2b', 'llama-3.2-3b', 'phi-3-mini', ...]

        >>> models = list_models(filter_by_context=100000, sort_by="memory")
        >>> print(models)
        ['phi-3-mini-128k']
    """
    all_models = list_available_models()

    # Apply size filter
    if filter_by_size:
        by_size = get_models_by_size()
        size_category = filter_by_size.lower()
        if size_category in by_size:
            all_models = [m for m in all_models if m in by_size[size_category]]
        else:
            logger.warning(f"Unknown size category: {filter_by_size}")

    # Apply context filter
    if filter_by_context:
        all_models = [
            m
            for m in all_models
            if get_model_spec(m)
            and get_model_spec(m)["context_length"] >= filter_by_context
        ]

    # Sort
    if sort_by == "memory":
        all_models.sort(
            key=lambda m: get_model_spec(m)["memory_estimate_mb"]
            if get_model_spec(m)
            else 0
        )
    elif sort_by == "context":
        all_models.sort(
            key=lambda m: get_model_spec(m)["context_length"]
            if get_model_spec(m)
            else 0,
            reverse=True,
        )
    else:  # sort by name (default)
        all_models.sort()

    return all_models


def get_model_info(model_name: str) -> ModelMetadata | None:
    """
    Get comprehensive metadata for a specific model.

    Args:
        model_name: Model identifier

    Returns:
        ModelMetadata dictionary with all model information, or None if not found

    Example:
        >>> info = get_model_info("phi-3-mini")
        >>> print(f"{info['name']}: {info['description']}")
        "phi-3-mini: Microsoft Phi-3 Mini (3.8B) - Fast and efficient, great for chat"
        >>> print(f"Memory: {info['memory_estimate_gb']:.1f}GB")
        "Memory: 2.4GB"
    """
    spec = get_model_spec(model_name)

    if spec is None:
        logger.warning(f"Model '{model_name}' not found in registry")
        return None

    # Determine size category
    mem_mb = spec["memory_estimate_mb"]
    if mem_mb < 3000:
        size_category = "small"
    elif mem_mb < 5000:
        size_category = "medium"
    else:
        size_category = "large"

    metadata: ModelMetadata = {
        "name": model_name,
        "path": spec["path"],
        "context_length": spec["context_length"],
        "quantization": spec["quantization"],
        "memory_estimate_mb": mem_mb,
        "memory_estimate_gb": round(mem_mb / 1024, 1),
        "description": spec["description"],
        "context_tokens_k": spec["context_length"] // 1024,
        "size_category": size_category,
    }

    return metadata


def download_model(model_name: str, verbose: bool = True) -> tuple[bool, str]:
    """
    Download a model from HuggingFace via mlx-lm.

    This function triggers the model download by attempting to load it.
    MLX will automatically download from HuggingFace on first use.

    Args:
        model_name: Model identifier to download
        verbose: Whether to show download progress (default: True)

    Returns:
        Tuple of (success, message)
        - success: True if download succeeded
        - message: Status or error message

    Example:
        >>> success, msg = download_model("phi-3-mini")
        >>> print(msg)
        "✓ Model 'phi-3-mini' downloaded successfully from mlx-community/Phi-3-mini-4k-instruct-4bit"

    Note:
        This function does NOT keep the model loaded in memory.
        It downloads and then immediately unloads to free memory.
    """
    from .loader import ModelLoader
    from .exceptions import ModelLoadError

    logger.info(f"Attempting to download model: {model_name}")

    # Validate model exists
    spec = get_model_spec(model_name)
    if spec is None:
        error_msg = f"✗ Model '{model_name}' not found in registry"
        logger.error(error_msg)
        return (False, error_msg)

    model_path = spec["path"]

    try:
        # Create a minimal config-like object
        class DownloadConfig:
            def __init__(self):
                self.model_name = model_name
                self.quantization = spec["quantization"]
                self.device = "auto"
                self.max_model_len = spec["context_length"]
                self.lazy_load = True
                self.warmup = False  # No warmup for download
                self.model_path = None

        config = DownloadConfig()

        # Create loader (will download if not cached)
        if verbose:
            logger.info(f"Downloading from {model_path}...")

        loader = ModelLoader(config)

        # Load model (triggers download)
        loader.load()

        # Immediately unload to free memory
        loader.unload()

        success_msg = (
            f"✓ Model '{model_name}' downloaded successfully from {model_path}"
        )
        logger.info(success_msg)
        return (True, success_msg)

    except ModelLoadError as e:
        error_msg = f"✗ Failed to download model: {str(e)}"
        logger.error(error_msg)
        return (False, error_msg)

    except Exception as e:
        error_msg = f"✗ Unexpected error during download: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (False, error_msg)


def compare_models(model_names: list[str]) -> str:
    """
    Compare multiple models side-by-side.

    Args:
        model_names: List of model identifiers to compare

    Returns:
        Formatted comparison table as string

    Example:
        >>> comparison = compare_models(["phi-3-mini", "mistral-7b", "llama-3-8b"])
        >>> print(comparison)
        Model Comparison
        ================
        Model          Size   Context  Description
        -------------- ------ -------- -----------
        phi-3-mini     2.4GB  4K       Microsoft Phi-3 Mini...
        mistral-7b     4.4GB  8K       Mistral 7B Instruct...
        llama-3-8b     4.9GB  8K       Meta Llama 3 8B...
    """
    lines = [
        "Model Comparison",
        "=" * 80,
        "",
        f"{'Model':<20} {'Size':<8} {'Context':<10} {'Description'}",
        "-" * 80,
    ]

    for model_name in model_names:
        info = get_model_info(model_name)

        if info is None:
            lines.append(f"{model_name:<20} NOT FOUND")
            continue

        size = f"{info['memory_estimate_gb']}GB"
        context = f"{info['context_tokens_k']}K"
        description = info["description"]

        # Truncate description if too long
        if len(description) > 40:
            description = description[:37] + "..."

        lines.append(f"{model_name:<20} {size:<8} {context:<10} {description}")

    return "\n".join(lines)


def find_models_by_description(search_term: str) -> list[str]:
    """
    Search for models by description keywords.

    Args:
        search_term: Keyword to search for (case-insensitive)

    Returns:
        List of model names with matching descriptions

    Example:
        >>> models = find_models_by_description("multilingual")
        >>> print(models)
        ['qwen-2.5-7b', 'qwen-2.5-3b']

        >>> models = find_models_by_description("long context")
        >>> print(models)
        ['phi-3-mini-128k', 'qwen-2.5-7b', 'qwen-2.5-3b']
    """
    search_lower = search_term.lower()
    matches = []

    for model_name, spec in MLX_SUPPORTED_MODELS.items():
        description = spec["description"].lower()
        if search_lower in description:
            matches.append(model_name)

    return sorted(matches)


def get_model_recommendation(
    use_case: str = "chat",
    max_memory_gb: float | None = None,
    min_context_k: int | None = None,
) -> tuple[str, str] | None:
    """
    Get personalized model recommendation based on constraints.

    Args:
        use_case: Use case identifier (chat, long_context, general, etc.)
        max_memory_gb: Maximum available memory in GB (optional)
        min_context_k: Minimum required context length in thousands of tokens (optional)

    Returns:
        Tuple of (model_name, reason) or None if no model meets criteria

    Example:
        >>> model, reason = get_model_recommendation("chat", max_memory_gb=3.0)
        >>> print(f"{model}: {reason}")
        "phi-3-mini: Fits in 3.0GB budget, optimized for chat"

        >>> model, reason = get_model_recommendation(
        ...     "general", max_memory_gb=6.0, min_context_k=32
        ... )
        >>> print(f"{model}: {reason}")
        "qwen-2.5-3b: 32K context, fits in 6.0GB budget"
    """
    # Start with use case recommendation
    base_recommendation = get_recommended_model(use_case)

    # Get all models
    candidates = list_available_models()

    # Apply memory constraint
    if max_memory_gb is not None:
        max_memory_mb = int(max_memory_gb * 1024)
        candidates = [
            m
            for m in candidates
            if get_model_spec(m)
            and get_model_spec(m)["memory_estimate_mb"] <= max_memory_mb
        ]

    # Apply context constraint
    if min_context_k is not None:
        min_context_tokens = min_context_k * 1024
        candidates = [
            m
            for m in candidates
            if get_model_spec(m)
            and get_model_spec(m)["context_length"] >= min_context_tokens
        ]

    # If no candidates meet criteria
    if not candidates:
        return None

    # Prefer base recommendation if it meets criteria
    if base_recommendation in candidates:
        spec = get_model_spec(base_recommendation)
        reason = f"Optimized for {use_case}"
        if max_memory_gb:
            reason += f", fits in {max_memory_gb}GB budget"
        if min_context_k:
            ctx_k = spec["context_length"] // 1024
            reason += f", {ctx_k}K context"
        return (base_recommendation, reason)

    # Otherwise, pick best candidate (smallest memory footprint with required features)
    candidates.sort(
        key=lambda m: get_model_spec(m)["memory_estimate_mb"]
        if get_model_spec(m)
        else float("inf")
    )

    best = candidates[0]
    spec = get_model_spec(best)

    reason = f"Best match for {use_case}"
    if max_memory_gb:
        mem_gb = spec["memory_estimate_mb"] / 1024
        reason += f", {mem_gb:.1f}GB fits in {max_memory_gb}GB budget"
    if min_context_k:
        ctx_k = spec["context_length"] // 1024
        reason += f", {ctx_k}K context meets {min_context_k}K requirement"

    return (best, reason)


def format_model_info(model_name: str) -> str:
    """
    Format detailed model information for display.

    Args:
        model_name: Model identifier

    Returns:
        Formatted multi-line string with model details

    Example:
        >>> print(format_model_info("phi-3-mini"))
        Model: phi-3-mini
        =================
        Description: Microsoft Phi-3 Mini (3.8B) - Fast and efficient, great for chat
        Repository:  mlx-community/Phi-3-mini-4k-instruct-4bit
        Size:        ~2.4GB (small)
        Context:     4096 tokens (4K)
        Quantization: 4bit
    """
    info = get_model_info(model_name)

    if info is None:
        return f"Model '{model_name}' not found in registry"

    lines = [
        f"Model: {info['name']}",
        "=" * (7 + len(info["name"])),
        "",
        f"Description:  {info['description']}",
        f"Repository:   {info['path']}",
        f"Size:         ~{info['memory_estimate_gb']}GB ({info['size_category']})",
        f"Context:      {info['context_length']} tokens ({info['context_tokens_k']}K)",
        f"Quantization: {info['quantization']}",
    ]

    return "\n".join(lines)


def check_model_compatibility(model_name: str) -> tuple[bool, str]:
    """
    Check if a model is compatible with current system.

    Args:
        model_name: Model identifier

    Returns:
        Tuple of (is_compatible, message)
        - is_compatible: True if model can be loaded
        - message: Compatibility status or issues

    Example:
        >>> is_compatible, msg = check_model_compatibility("phi-3-mini")
        >>> print(msg)
        "✓ Model compatible: phi-3-mini requires ~2500MB, 16384MB available"
    """
    from ..devices import DeviceDetector

    # Check if model exists
    spec = get_model_spec(model_name)
    if spec is None:
        return (False, f"✗ Model '{model_name}' not found in registry")

    # Check platform
    is_valid, platform_msg = DeviceDetector.validate_platform()
    if not is_valid:
        return (False, f"✗ Platform incompatible: {platform_msg}")

    # Check memory
    can_fit, mem_msg = MemoryManager.can_fit_model(
        model_name, spec["quantization"], "mps"
    )

    if can_fit:
        return (True, f"✓ Model compatible: {mem_msg}")
    else:
        return (False, f"✗ Model incompatible: {mem_msg}")
