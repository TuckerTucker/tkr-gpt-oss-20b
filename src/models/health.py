"""
Health check utilities for MLX model validation.

Provides functions to verify model loading status, Metal backend availability,
and overall model health for reliable inference operations.
"""

import logging
from typing import TypedDict
from .exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class HealthStatus(TypedDict):
    """Health status report for model and device."""

    model_loaded: bool
    model_name: str | None
    metal_available: bool
    metal_active_memory_mb: int
    metal_peak_memory_mb: int
    metal_cache_memory_mb: int
    device: str
    platform_valid: bool
    can_generate: bool
    overall_healthy: bool
    issues: list[str]


def check_model_loaded(loader) -> tuple[bool, str]:
    """
    Verify that a model is loaded and ready for inference.

    Args:
        loader: ModelLoader instance to check

    Returns:
        Tuple of (is_loaded, message)
        - is_loaded: True if model is loaded and ready
        - message: Status message

    Example:
        >>> from src.models import ModelLoader
        >>> from src.config import ModelConfig
        >>> loader = ModelLoader(ModelConfig(model_name="phi-3-mini"))
        >>> is_loaded, msg = check_model_loaded(loader)
        >>> print(msg)
        "Model not loaded"
    """
    try:
        is_loaded = loader.is_loaded()

        if is_loaded:
            model_info = loader.get_info()
            return (
                True,
                f"✓ Model '{model_info.model_name}' loaded on {model_info.device}, "
                f"using {model_info.memory_usage_mb}MB",
            )
        else:
            return (False, "Model not loaded")

    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        return (False, f"Error checking model: {str(e)}")


def check_metal_backend() -> tuple[bool, str]:
    """
    Verify Metal GPU backend is available and functional.

    Returns:
        Tuple of (is_available, message)
        - is_available: True if Metal is available and working
        - message: Status message with details

    Example:
        >>> is_available, msg = check_metal_backend()
        >>> print(msg)
        "✓ Metal GPU available - 16384MB active, 20480MB peak"
    """
    try:
        from ..devices import DeviceDetector

        # Check if platform supports Metal
        is_valid, platform_msg = DeviceDetector.validate_platform()

        if not is_valid:
            return (False, f"✗ Metal not available: {platform_msg}")

        # Get Metal memory info
        metal_info = DeviceDetector.get_metal_info()

        if not metal_info or not metal_info.get("metal_available", False):
            error = metal_info.get("error", "Unknown error") if metal_info else "MLX not installed"
            return (False, f"✗ Metal backend unavailable: {error}")

        # Format memory stats
        active_mb = int(metal_info.get("active_memory_mb", 0))
        peak_mb = int(metal_info.get("peak_memory_mb", 0))
        cache_mb = int(metal_info.get("cache_memory_mb", 0))

        return (
            True,
            f"✓ Metal GPU available - {active_mb}MB active, {peak_mb}MB peak, {cache_mb}MB cache",
        )

    except ImportError as e:
        return (False, f"✗ Metal unavailable: MLX not installed ({e})")
    except Exception as e:
        logger.error(f"Error checking Metal backend: {e}")
        return (False, f"✗ Error checking Metal: {str(e)}")


def get_health_status(loader) -> HealthStatus:
    """
    Get comprehensive health status report for model and device.

    Args:
        loader: ModelLoader instance to check

    Returns:
        HealthStatus dictionary with detailed status information

    Example:
        >>> from src.models import ModelLoader
        >>> from src.config import ModelConfig
        >>> loader = ModelLoader(ModelConfig(model_name="phi-3-mini"))
        >>> loader.load()
        >>> status = get_health_status(loader)
        >>> print(f"Healthy: {status['overall_healthy']}")
        >>> for issue in status['issues']:
        ...     print(f"  - {issue}")
    """
    from ..devices import DeviceDetector
    from .memory import MemoryManager

    status: HealthStatus = {
        "model_loaded": False,
        "model_name": None,
        "metal_available": False,
        "metal_active_memory_mb": 0,
        "metal_peak_memory_mb": 0,
        "metal_cache_memory_mb": 0,
        "device": "unknown",
        "platform_valid": False,
        "can_generate": False,
        "overall_healthy": False,
        "issues": [],
    }

    # Check platform
    is_valid, platform_msg = DeviceDetector.validate_platform()
    status["platform_valid"] = is_valid

    if not is_valid:
        status["issues"].append(f"Platform invalid: {platform_msg}")

    # Check device
    device_info = DeviceDetector.get_device_info()
    status["device"] = device_info.get("device", "unknown")

    # Check Metal
    metal_info = DeviceDetector.get_metal_info()
    if metal_info and metal_info.get("metal_available", False):
        status["metal_available"] = True
        status["metal_active_memory_mb"] = int(metal_info.get("active_memory_mb", 0))
        status["metal_peak_memory_mb"] = int(metal_info.get("peak_memory_mb", 0))
        status["metal_cache_memory_mb"] = int(metal_info.get("cache_memory_mb", 0))
    else:
        error = metal_info.get("error", "Unknown") if metal_info else "Not available"
        status["issues"].append(f"Metal unavailable: {error}")

    # Check model
    try:
        model_loaded = loader.is_loaded()
        status["model_loaded"] = model_loaded

        if model_loaded:
            model_info = loader.get_info()
            status["model_name"] = model_info.model_name

            # Try health check (generation test)
            try:
                can_generate = loader.health_check()
                status["can_generate"] = can_generate

                if not can_generate:
                    status["issues"].append("Model failed health check - cannot generate")

            except Exception as e:
                status["issues"].append(f"Health check failed: {str(e)}")

        else:
            status["issues"].append("Model not loaded")

    except Exception as e:
        status["issues"].append(f"Error checking model: {str(e)}")

    # Overall health: platform valid, Metal available, model loaded and can generate
    status["overall_healthy"] = (
        status["platform_valid"]
        and status["metal_available"]
        and status["model_loaded"]
        and status["can_generate"]
        and len(status["issues"]) == 0
    )

    logger.debug(f"Health status: {status}")
    return status


def format_health_report(status: HealthStatus) -> str:
    """
    Format health status as human-readable report.

    Args:
        status: HealthStatus dictionary from get_health_status()

    Returns:
        Formatted multi-line string report

    Example:
        >>> status = get_health_status(loader)
        >>> print(format_health_report(status))
        System Health Report
        ====================
        Overall Status: ✓ Healthy
        ...
    """
    lines = [
        "System Health Report",
        "=" * 50,
        "",
    ]

    # Overall status
    if status["overall_healthy"]:
        lines.append("Overall Status: ✓ Healthy")
    else:
        lines.append("Overall Status: ✗ Unhealthy")

    lines.append("")

    # Platform
    platform_icon = "✓" if status["platform_valid"] else "✗"
    lines.append(f"{platform_icon} Platform: {status['device']}")

    # Metal
    metal_icon = "✓" if status["metal_available"] else "✗"
    if status["metal_available"]:
        lines.append(
            f"{metal_icon} Metal GPU: "
            f"{status['metal_active_memory_mb']}MB active, "
            f"{status['metal_peak_memory_mb']}MB peak, "
            f"{status['metal_cache_memory_mb']}MB cache"
        )
    else:
        lines.append(f"{metal_icon} Metal GPU: Not available")

    # Model
    model_icon = "✓" if status["model_loaded"] else "✗"
    if status["model_loaded"]:
        lines.append(f"{model_icon} Model: {status['model_name']} loaded")
    else:
        lines.append(f"{model_icon} Model: Not loaded")

    # Generation capability
    gen_icon = "✓" if status["can_generate"] else "✗"
    lines.append(f"{gen_icon} Generation: {'Ready' if status['can_generate'] else 'Not ready'}")

    # Issues
    if status["issues"]:
        lines.append("")
        lines.append("Issues:")
        for issue in status["issues"]:
            lines.append(f"  • {issue}")

    return "\n".join(lines)
