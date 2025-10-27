"""
Model registry for MLX-supported models.

Catalog of pre-quantized models from mlx-community on HuggingFace,
optimized for Apple Silicon with Metal acceleration.
"""

import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


class ModelSpec(TypedDict):
    """Model specification with metadata."""

    path: str  # HuggingFace repository path
    context_length: int  # Maximum context window in tokens
    quantization: str  # Quantization level (4bit, 8bit, fp16)
    memory_estimate_mb: int  # Estimated memory requirement
    description: str  # Human-readable description


# Registry of MLX-optimized models from mlx-community
# All models are 4-bit quantized for efficient inference on Apple Silicon
MLX_SUPPORTED_MODELS: dict[str, ModelSpec] = {
    "phi-3-mini": {
        "path": "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "context_length": 4096,
        "quantization": "4bit",
        "memory_estimate_mb": 2500,
        "description": "Microsoft Phi-3 Mini (3.8B) - Fast and efficient, great for chat",
    },
    "phi-3-mini-128k": {
        "path": "mlx-community/Phi-3-mini-128k-instruct-4bit",
        "context_length": 128000,
        "quantization": "4bit",
        "memory_estimate_mb": 2700,
        "description": "Microsoft Phi-3 Mini with 128K context - Ideal for long conversations",
    },
    "mistral-7b": {
        "path": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 4500,
        "description": "Mistral 7B Instruct v0.3 - Excellent general-purpose model",
    },
    "mistral-7b-v0.2": {
        "path": "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 4500,
        "description": "Mistral 7B Instruct v0.2 - Stable previous version",
    },
    "llama-3-8b": {
        "path": "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 5000,
        "description": "Meta Llama 3 8B Instruct - High quality instruction-following",
    },
    "llama-3.1-8b": {
        "path": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 5000,
        "description": "Meta Llama 3.1 8B Instruct - Latest version with improvements",
    },
    "llama-3.2-3b": {
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 2200,
        "description": "Meta Llama 3.2 3B Instruct - Lightweight and fast",
    },
    "gemma-2-9b": {
        "path": "mlx-community/gemma-2-9b-it-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 5500,
        "description": "Google Gemma 2 9B Instruct - Strong performance across tasks",
    },
    "gemma-2-2b": {
        "path": "mlx-community/gemma-2-2b-it-4bit",
        "context_length": 8192,
        "quantization": "4bit",
        "memory_estimate_mb": 1800,
        "description": "Google Gemma 2 2B Instruct - Very lightweight",
    },
    "qwen-2.5-7b": {
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "context_length": 32768,
        "quantization": "4bit",
        "memory_estimate_mb": 4500,
        "description": "Qwen 2.5 7B Instruct - Multilingual with long context",
    },
    "qwen-2.5-3b": {
        "path": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "context_length": 32768,
        "quantization": "4bit",
        "memory_estimate_mb": 2300,
        "description": "Qwen 2.5 3B Instruct - Efficient multilingual model",
    },
}


def get_model_spec(model_name: str) -> ModelSpec | None:
    """
    Get model specification by name.

    Args:
        model_name: Model identifier (case-insensitive)

    Returns:
        ModelSpec if found, None otherwise
    """
    # Try exact match first (case-insensitive)
    model_lower = model_name.lower()
    if model_lower in MLX_SUPPORTED_MODELS:
        return MLX_SUPPORTED_MODELS[model_lower]

    # Try partial match
    for key, spec in MLX_SUPPORTED_MODELS.items():
        if model_lower in key or key in model_lower:
            logger.debug(f"Matched '{model_name}' to '{key}'")
            return spec

    return None


def list_available_models() -> list[str]:
    """
    List all available model identifiers.

    Returns:
        List of model names that can be loaded
    """
    return sorted(MLX_SUPPORTED_MODELS.keys())


def get_models_by_size() -> dict[str, list[str]]:
    """
    Group models by approximate size category.

    Returns:
        Dictionary with size categories and model lists
    """
    small = []  # < 3GB
    medium = []  # 3-5GB
    large = []  # > 5GB

    for name, spec in MLX_SUPPORTED_MODELS.items():
        mem = spec["memory_estimate_mb"]
        if mem < 3000:
            small.append(name)
        elif mem < 5000:
            medium.append(name)
        else:
            large.append(name)

    return {
        "small": sorted(small),
        "medium": sorted(medium),
        "large": sorted(large),
    }


def format_model_list() -> str:
    """
    Format model list for display.

    Returns:
        Formatted string listing all available models with descriptions
    """
    lines = ["Available MLX Models:", ""]

    by_size = get_models_by_size()

    for category, models in [
        ("Small Models (< 3GB)", by_size["small"]),
        ("Medium Models (3-5GB)", by_size["medium"]),
        ("Large Models (> 5GB)", by_size["large"]),
    ]:
        if not models:
            continue

        lines.append(f"{category}:")
        for model_name in models:
            spec = MLX_SUPPORTED_MODELS[model_name]
            mem_gb = spec["memory_estimate_mb"] / 1024
            ctx_k = spec["context_length"] // 1024
            lines.append(
                f"  • {model_name:20s} - {spec['description']} "
                f"(~{mem_gb:.1f}GB, {ctx_k}K context)"
            )
        lines.append("")

    return "\n".join(lines)


def validate_model_name(model_name: str) -> tuple[bool, str]:
    """
    Validate that a model name is supported.

    Args:
        model_name: Model identifier to validate

    Returns:
        Tuple of (is_valid, message)
        - is_valid: True if model is supported
        - message: Explanation or suggestion
    """
    spec = get_model_spec(model_name)

    if spec is not None:
        return (True, f"Model '{model_name}' is supported")

    # Provide helpful suggestions
    available = list_available_models()
    suggestion = ""

    # Try to find similar names
    model_lower = model_name.lower()
    similar = [m for m in available if model_lower in m or m in model_lower]

    if similar:
        suggestion = f" Did you mean: {', '.join(similar[:3])}?"

    return (
        False,
        f"Model '{model_name}' not found in registry.{suggestion} "
        f"Run 'list-models' to see all available models."
    )


# Default model for new users
DEFAULT_MODEL = "phi-3-mini"

# Recommended models by use case
RECOMMENDED_MODELS = {
    "chat": "phi-3-mini",  # Fast, efficient, great quality
    "long_context": "phi-3-mini-128k",  # 128K context window
    "general": "mistral-7b",  # Excellent all-around performance
    "lightweight": "gemma-2-2b",  # Minimal memory footprint
    "multilingual": "qwen-2.5-7b",  # Best for non-English
    "quality": "llama-3.1-8b",  # Highest quality at reasonable size
}


def get_recommended_model(use_case: str = "chat") -> str:
    """
    Get recommended model for a use case.

    Args:
        use_case: Use case identifier (chat, long_context, general, etc.)

    Returns:
        Recommended model name
    """
    return RECOMMENDED_MODELS.get(use_case.lower(), DEFAULT_MODEL)
