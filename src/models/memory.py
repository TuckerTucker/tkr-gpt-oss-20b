"""
Memory management for MLX models on Apple Silicon.

Provides utilities to track Metal memory usage, estimate model memory
requirements, and validate that models will fit in available memory.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Track and report Metal memory usage for MLX models.

    MLX uses unified memory on Apple Silicon, where GPU and CPU share
    the same memory pool. This manager tracks Metal-specific allocations.
    """

    @staticmethod
    def get_model_memory_usage() -> int:
        """
        Get current Metal memory usage in MB.

        Returns:
            Active Metal memory usage in megabytes

        Note:
            Returns 0 if MLX is not available or on unsupported platforms
        """
        try:
            import mlx.core as mx

            active_memory_bytes = mx.metal.get_active_memory()
            active_memory_mb = int(active_memory_bytes / (1024 * 1024))

            logger.debug(f"Metal active memory: {active_memory_mb}MB")
            return active_memory_mb

        except ImportError:
            logger.debug("MLX not available - cannot get memory usage")
            return 0
        except Exception as e:
            logger.warning(f"Error getting Metal memory usage: {e}")
            return 0

    @staticmethod
    def get_peak_memory_usage() -> int:
        """
        Get peak Metal memory usage in MB.

        Returns:
            Peak Metal memory usage since process start, in megabytes
        """
        try:
            import mlx.core as mx

            peak_memory_bytes = mx.metal.get_peak_memory()
            peak_memory_mb = int(peak_memory_bytes / (1024 * 1024))

            logger.debug(f"Metal peak memory: {peak_memory_mb}MB")
            return peak_memory_mb

        except ImportError:
            return 0
        except Exception as e:
            logger.warning(f"Error getting peak memory usage: {e}")
            return 0

    @staticmethod
    def get_cache_memory_usage() -> int:
        """
        Get Metal cache memory usage in MB.

        Returns:
            Metal cache memory in megabytes
        """
        try:
            import mlx.core as mx

            cache_memory_bytes = mx.metal.get_cache_memory()
            cache_memory_mb = int(cache_memory_bytes / (1024 * 1024))

            logger.debug(f"Metal cache memory: {cache_memory_mb}MB")
            return cache_memory_mb

        except ImportError:
            return 0
        except Exception as e:
            logger.warning(f"Error getting cache memory: {e}")
            return 0

    @staticmethod
    def get_memory_stats() -> dict:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with:
            - active_mb: Currently active memory
            - peak_mb: Peak memory usage
            - cache_mb: Cache memory
            - total_allocated_mb: Total (active + cache)
        """
        active = MemoryManager.get_model_memory_usage()
        peak = MemoryManager.get_peak_memory_usage()
        cache = MemoryManager.get_cache_memory_usage()

        stats = {
            "active_mb": active,
            "peak_mb": peak,
            "cache_mb": cache,
            "total_allocated_mb": active + cache,
        }

        logger.debug(f"Memory stats: {stats}")
        return stats

    @staticmethod
    def estimate_memory_requirement(model_name: str, quantization: str) -> int:
        """
        Estimate memory needed for a model with given quantization.

        Args:
            model_name: Model identifier (e.g., "phi-3-mini", "mistral-7b")
            quantization: Quantization level ("4bit", "8bit", "fp16")

        Returns:
            Estimated memory requirement in MB

        Note:
            These are conservative estimates. Actual usage may vary based on
            context length and other runtime factors. Includes ~20% overhead
            for KV cache and runtime buffers.
        """
        # Base parameter counts (in billions)
        param_counts = {
            "phi-3-mini": 3.8,  # 3.8B parameters
            "phi-3": 3.8,
            "mistral-7b": 7.2,  # 7.2B parameters
            "mistral": 7.2,
            "llama-3-8b": 8.0,  # 8B parameters
            "llama-3": 8.0,
            "llama": 8.0,
        }

        # Find matching model (case-insensitive, partial match)
        model_lower = model_name.lower()
        params = None
        for key, value in param_counts.items():
            if key in model_lower:
                params = value
                break

        # Default to 7B if unknown
        if params is None:
            logger.warning(
                f"Unknown model '{model_name}', estimating for 7B parameters"
            )
            params = 7.0

        # Memory per parameter in bytes based on quantization
        bytes_per_param = {
            "4bit": 0.5,  # 4 bits = 0.5 bytes
            "8bit": 1.0,  # 8 bits = 1 byte
            "fp16": 2.0,  # 16 bits = 2 bytes
            "fp32": 4.0,  # 32 bits = 4 bytes
        }

        # Normalize quantization string
        quant_lower = quantization.lower()
        if "4" in quant_lower:
            bytes_pp = bytes_per_param["4bit"]
        elif "8" in quant_lower:
            bytes_pp = bytes_per_param["8bit"]
        elif "16" in quant_lower or "fp16" in quant_lower:
            bytes_pp = bytes_per_param["fp16"]
        elif "32" in quant_lower or "fp32" in quant_lower:
            bytes_pp = bytes_per_param["fp32"]
        else:
            logger.warning(f"Unknown quantization '{quantization}', assuming 4bit")
            bytes_pp = bytes_per_param["4bit"]

        # Calculate base memory requirement
        # params in billions, bytes_pp is bytes per parameter
        base_memory_gb = params * bytes_pp

        # Add 20% overhead for KV cache, activations, and runtime buffers
        total_memory_gb = base_memory_gb * 1.2

        # Convert to MB
        total_memory_mb = int(total_memory_gb * 1024)

        logger.debug(
            f"Estimated {model_name} ({params}B params) with {quantization}: "
            f"{total_memory_mb}MB"
        )

        return total_memory_mb

    @staticmethod
    def get_available_memory() -> Optional[int]:
        """
        Get available system memory in MB.

        Returns:
            Available memory in MB, or None if cannot determine

        Note:
            On Apple Silicon with unified memory, this returns the
            available system RAM that could be used by Metal.
        """
        try:
            import psutil

            available_bytes = psutil.virtual_memory().available
            available_mb = int(available_bytes / (1024 * 1024))

            logger.debug(f"Available system memory: {available_mb}MB")
            return available_mb

        except ImportError:
            logger.debug("psutil not available - cannot determine available memory")
            return None
        except Exception as e:
            logger.warning(f"Error getting available memory: {e}")
            return None

    @staticmethod
    def can_fit_model(model_name: str, quantization: str, device: str) -> tuple[bool, str]:
        """
        Check if model will fit in available memory.

        Args:
            model_name: Model identifier
            quantization: Quantization level
            device: Target device ("mps" or "cpu")

        Returns:
            Tuple of (can_fit, message)
            - can_fit: True if model should fit
            - message: Explanation of the decision
        """
        required_mb = MemoryManager.estimate_memory_requirement(
            model_name, quantization
        )

        # Get available memory
        available_mb = MemoryManager.get_available_memory()

        if available_mb is None:
            return (
                True,
                f"Cannot determine available memory. Model requires ~{required_mb}MB. "
                f"Proceeding with load attempt."
            )

        # Check if it fits (leave 2GB buffer for system)
        buffer_mb = 2048
        usable_mb = available_mb - buffer_mb

        if required_mb <= usable_mb:
            return (
                True,
                f"Model should fit: requires ~{required_mb}MB, "
                f"have {usable_mb}MB available (after {buffer_mb}MB system buffer)"
            )
        else:
            return (
                False,
                f"Model may not fit: requires ~{required_mb}MB, "
                f"only {usable_mb}MB available (after {buffer_mb}MB system buffer). "
                f"Try a more aggressive quantization or smaller model."
            )

    @staticmethod
    def clear_cache() -> None:
        """
        Clear Metal memory cache.

        Attempts to free cached memory allocations.
        """
        try:
            import mlx.core as mx

            mx.metal.clear_cache()
            logger.info("Metal memory cache cleared")

        except ImportError:
            logger.debug("MLX not available - cannot clear cache")
        except Exception as e:
            logger.warning(f"Error clearing Metal cache: {e}")

    @staticmethod
    def reset_peak_memory() -> None:
        """
        Reset peak memory tracking.

        Useful for benchmarking individual operations.
        """
        try:
            import mlx.core as mx

            mx.metal.reset_peak_memory()
            logger.debug("Peak memory tracking reset")

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error resetting peak memory: {e}")
