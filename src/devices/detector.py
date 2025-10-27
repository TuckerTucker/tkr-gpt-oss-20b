"""
Device detection for Apple Silicon with MLX support.

Validates platform compatibility and provides device information for MLX-based
model loading on Apple Silicon (M1/M2/M3/M4) processors.
"""

import platform
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class DeviceDetector:
    """
    Auto-detect Apple Silicon device for MLX.

    MLX is designed specifically for Apple Silicon and requires:
    - ARM64 architecture (Apple Silicon M1/M2/M3/M4)
    - macOS operating system
    - Metal support (implicit on Apple Silicon)
    """

    @staticmethod
    def detect() -> Literal["mps", "cpu"]:
        """
        Detect best available device for MLX.

        Returns:
            Device name ("mps" for Metal/Apple Silicon, "cpu" as fallback)

        Note:
            MLX requires Apple Silicon. On non-Apple Silicon systems,
            this returns "cpu" but MLX will not function properly.
        """
        if DeviceDetector.is_apple_silicon():
            logger.info("✓ Apple Silicon detected - MLX can use Metal acceleration")
            return "mps"
        else:
            logger.warning(
                "⚠️  Apple Silicon not detected - MLX requires Apple Silicon (M1/M2/M3/M4)"
            )
            return "cpu"

    @staticmethod
    def is_apple_silicon() -> bool:
        """
        Check if running on Apple Silicon.

        Returns:
            True if on Apple Silicon (arm64), False otherwise
        """
        try:
            machine = platform.machine().lower()
            system = platform.system().lower()

            # Apple Silicon is identified by arm64 architecture on Darwin (macOS)
            is_arm64 = machine in ["arm64", "aarch64"]
            is_macos = system == "darwin"

            result = is_arm64 and is_macos

            if result:
                logger.debug(f"Platform check: {machine} on {system} → Apple Silicon")
            else:
                logger.debug(f"Platform check: {machine} on {system} → Not Apple Silicon")

            return result

        except Exception as e:
            logger.error(f"Error detecting platform: {e}")
            return False

    @staticmethod
    def validate_platform() -> tuple[bool, str]:
        """
        Validate that the platform supports MLX.

        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if platform supports MLX
            - message: Descriptive message about platform compatibility
        """
        if not DeviceDetector.is_apple_silicon():
            machine = platform.machine()
            system = platform.system()
            return (
                False,
                f"MLX requires Apple Silicon (M1/M2/M3/M4). "
                f"Current platform: {machine} on {system}"
            )

        return (True, "Platform validated: Apple Silicon with Metal support")

    @staticmethod
    def get_device_info() -> dict:
        """
        Get detailed device information.

        Returns:
            Dictionary with device information:
            - machine: Architecture (e.g., "arm64")
            - system: Operating system (e.g., "Darwin")
            - platform: Full platform string
            - processor: Processor name
            - is_apple_silicon: Whether on Apple Silicon
            - device: Recommended device for MLX ("mps" or "cpu")
        """
        try:
            info = {
                "machine": platform.machine(),
                "system": platform.system(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "is_apple_silicon": DeviceDetector.is_apple_silicon(),
                "device": DeviceDetector.detect(),
            }

            logger.debug(f"Device info: {info}")
            return info

        except Exception as e:
            logger.error(f"Error gathering device info: {e}")
            return {
                "machine": "unknown",
                "system": "unknown",
                "platform": "unknown",
                "processor": "unknown",
                "is_apple_silicon": False,
                "device": "cpu",
            }

    @staticmethod
    def get_metal_info() -> dict:
        """
        Get Metal GPU information if available.

        Returns:
            Dictionary with Metal information or empty dict if unavailable
        """
        if not DeviceDetector.is_apple_silicon():
            return {}

        try:
            # Try to get Metal device info through MLX
            import mlx.core as mx

            info = {
                "metal_available": True,
                "active_memory_mb": mx.metal.get_active_memory() / (1024 * 1024),
                "peak_memory_mb": mx.metal.get_peak_memory() / (1024 * 1024),
                "cache_memory_mb": mx.metal.get_cache_memory() / (1024 * 1024),
            }

            logger.debug(f"Metal info: {info}")
            return info

        except ImportError:
            logger.debug("MLX not installed - Metal info unavailable")
            return {"metal_available": False, "error": "MLX not installed"}
        except Exception as e:
            logger.warning(f"Error getting Metal info: {e}")
            return {"metal_available": False, "error": str(e)}
