"""
Performance optimization utilities for inference.

This module provides optimization strategies for Metal acceleration,
memory management, and batch inference improvements.
"""

import os
import psutil
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """
    System memory statistics.

    Attributes:
        total_bytes: Total system memory
        available_bytes: Available memory
        used_bytes: Used memory
        percent_used: Percentage of memory used
        available_gb: Available memory in GB
        used_gb: Used memory in GB
    """
    total_bytes: int
    available_bytes: int
    used_bytes: int
    percent_used: float

    @property
    def available_gb(self) -> float:
        """Get available memory in GB."""
        return self.available_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Get used memory in GB."""
        return self.used_bytes / (1024 ** 3)

    def __str__(self) -> str:
        """Human-readable memory stats."""
        return (
            f"Memory: {self.used_gb:.2f}GB/{self.total_bytes / (1024**3):.2f}GB "
            f"({self.percent_used:.1f}% used, {self.available_gb:.2f}GB available)"
        )


class MemoryMonitor:
    """
    Monitor system memory usage.

    Provides utilities for checking available memory and making
    memory-aware decisions during inference.
    """

    @staticmethod
    def get_memory_stats() -> MemoryStats:
        """
        Get current system memory statistics.

        Returns:
            MemoryStats with current memory usage
        """
        vm = psutil.virtual_memory()
        return MemoryStats(
            total_bytes=vm.total,
            available_bytes=vm.available,
            used_bytes=vm.used,
            percent_used=vm.percent
        )

    @staticmethod
    def has_available_memory(required_gb: float) -> bool:
        """
        Check if required memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory is available
        """
        stats = MemoryMonitor.get_memory_stats()
        available = stats.available_gb
        has_memory = available >= required_gb

        logger.debug(
            f"Memory check: {available:.2f}GB available, "
            f"{required_gb:.2f}GB required - {'OK' if has_memory else 'INSUFFICIENT'}"
        )

        return has_memory

    @staticmethod
    def estimate_memory_for_generation(
        prompt_tokens: int,
        max_tokens: int,
        bytes_per_token: float = 4.0
    ) -> float:
        """
        Estimate memory required for generation.

        Args:
            prompt_tokens: Number of tokens in prompt
            max_tokens: Maximum tokens to generate
            bytes_per_token: Estimated bytes per token (default: 4 for fp32)

        Returns:
            Estimated memory in GB
        """
        total_tokens = prompt_tokens + max_tokens
        bytes_needed = total_tokens * bytes_per_token
        gb_needed = bytes_needed / (1024 ** 3)

        logger.debug(
            f"Estimated memory: {gb_needed:.3f}GB for {total_tokens} tokens"
        )

        return gb_needed


class MetalCacheManager:
    """
    Manage Metal GPU cache for optimal performance.

    Provides utilities for cache management on Apple Silicon devices
    using Metal for GPU acceleration.
    """

    def __init__(self):
        """Initialize Metal cache manager."""
        self._cache_enabled = self._check_metal_available()

    def _check_metal_available(self) -> bool:
        """
        Check if Metal acceleration is available.

        Returns:
            True if Metal is available
        """
        try:
            import mlx.core as mx
            # Check if Metal is available
            metal_available = mx.metal.is_available()
            if metal_available:
                logger.info("Metal acceleration detected")
            else:
                logger.warning("Metal acceleration not available")
            return metal_available
        except ImportError:
            logger.warning("MLX not installed, Metal acceleration unavailable")
            return False
        except Exception as e:
            logger.warning(f"Error checking Metal availability: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if Metal cache management is available.

        Returns:
            True if Metal is available
        """
        return self._cache_enabled

    def clear_cache(self) -> None:
        """
        Clear Metal GPU cache.

        This can help recover from memory issues or prepare for large operations.
        """
        if not self._cache_enabled:
            logger.debug("Metal not available, skipping cache clear")
            return

        try:
            import mlx.core as mx
            # Clear Metal cache
            mx.metal.clear_cache()
            logger.info("Metal cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear Metal cache: {e}")

    def optimize_for_generation(self, max_tokens: int) -> None:
        """
        Optimize Metal cache for upcoming generation.

        Args:
            max_tokens: Expected maximum tokens to generate
        """
        if not self._cache_enabled:
            return

        # For large generations, pre-clear cache
        if max_tokens > 1024:
            logger.debug("Large generation detected, clearing Metal cache")
            self.clear_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get Metal cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self._cache_enabled:
            return {"available": False}

        try:
            import mlx.core as mx
            # Get device info
            stats = {
                "available": True,
                "metal_available": mx.metal.is_available(),
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get Metal cache stats: {e}")
            return {"available": False, "error": str(e)}


class MemoryAwareGenerator:
    """
    Memory-aware generation parameter adjustment.

    Automatically adjusts generation parameters based on available memory
    to prevent out-of-memory errors.
    """

    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        safety_margin_gb: float = 2.0
    ):
        """
        Initialize memory-aware generator.

        Args:
            memory_monitor: Memory monitor instance
            safety_margin_gb: Safety margin to keep free (in GB)
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.safety_margin_gb = safety_margin_gb

    def adjust_max_tokens(
        self,
        sampling_params: SamplingParams,
        prompt_tokens: int
    ) -> SamplingParams:
        """
        Adjust max_tokens based on available memory.

        Args:
            sampling_params: Original sampling parameters
            prompt_tokens: Number of tokens in the prompt

        Returns:
            Adjusted SamplingParams with memory-safe max_tokens
        """
        stats = self.memory_monitor.get_memory_stats()
        available_gb = stats.available_gb - self.safety_margin_gb

        if available_gb <= 0:
            logger.warning("Insufficient available memory, using minimum max_tokens")
            return sampling_params.copy(max_tokens=32)

        # Estimate memory per token (conservative estimate)
        bytes_per_token = 8.0  # Conservative for fp16 with overhead

        # Calculate safe max_tokens
        available_bytes = available_gb * (1024 ** 3)
        safe_tokens = int(available_bytes / bytes_per_token)

        # Cap at original max_tokens if we have enough memory
        adjusted_max_tokens = min(sampling_params.max_tokens, safe_tokens)

        # Ensure minimum of 32 tokens
        adjusted_max_tokens = max(32, adjusted_max_tokens)

        if adjusted_max_tokens < sampling_params.max_tokens:
            logger.warning(
                f"Reducing max_tokens from {sampling_params.max_tokens} "
                f"to {adjusted_max_tokens} due to memory constraints "
                f"({available_gb:.2f}GB available)"
            )
            return sampling_params.copy(max_tokens=adjusted_max_tokens)

        return sampling_params

    def check_generation_feasible(
        self,
        prompt_tokens: int,
        max_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """
        Check if generation is feasible with current memory.

        Args:
            prompt_tokens: Number of tokens in prompt
            max_tokens: Requested max tokens to generate

        Returns:
            Tuple of (is_feasible, reason_if_not)
        """
        stats = self.memory_monitor.get_memory_stats()
        required_gb = MemoryMonitor.estimate_memory_for_generation(
            prompt_tokens, max_tokens
        )

        available_gb = stats.available_gb - self.safety_margin_gb

        if required_gb > available_gb:
            reason = (
                f"Insufficient memory: need {required_gb:.2f}GB, "
                f"have {available_gb:.2f}GB available"
            )
            return False, reason

        return True, None


class BatchOptimizer:
    """
    Optimize batch inference operations.

    Provides utilities for dynamic batch sizing and parallel processing
    optimization.
    """

    def __init__(self, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize batch optimizer.

        Args:
            memory_monitor: Memory monitor instance
        """
        self.memory_monitor = memory_monitor or MemoryMonitor()

    def calculate_optimal_batch_size(
        self,
        total_prompts: int,
        avg_prompt_tokens: int,
        max_tokens: int,
        max_batch_size: int = 32
    ) -> int:
        """
        Calculate optimal batch size based on memory.

        Args:
            total_prompts: Total number of prompts to process
            avg_prompt_tokens: Average tokens per prompt
            max_tokens: Maximum tokens to generate per prompt
            max_batch_size: Maximum batch size limit

        Returns:
            Optimal batch size
        """
        stats = self.memory_monitor.get_memory_stats()
        available_gb = stats.available_gb * 0.8  # Use 80% of available

        # Estimate memory per prompt
        tokens_per_prompt = avg_prompt_tokens + max_tokens
        gb_per_prompt = MemoryMonitor.estimate_memory_for_generation(
            avg_prompt_tokens, max_tokens
        )

        if gb_per_prompt <= 0:
            return 1

        # Calculate how many prompts fit in memory
        batch_size = int(available_gb / gb_per_prompt)

        # Apply constraints
        batch_size = max(1, min(batch_size, max_batch_size, total_prompts))

        logger.info(
            f"Calculated optimal batch size: {batch_size} "
            f"(available: {available_gb:.2f}GB, per-prompt: {gb_per_prompt:.3f}GB)"
        )

        return batch_size

    def should_use_parallel(
        self,
        batch_size: int,
        cpu_count: Optional[int] = None
    ) -> bool:
        """
        Determine if parallel processing should be used.

        Args:
            batch_size: Size of batch to process
            cpu_count: Number of CPUs (None = auto-detect)

        Returns:
            True if parallel processing is recommended
        """
        if cpu_count is None:
            cpu_count = os.cpu_count() or 1

        # Only use parallel for batches >= 4 and multiple CPUs
        use_parallel = batch_size >= 4 and cpu_count > 1

        logger.debug(
            f"Parallel processing: {'enabled' if use_parallel else 'disabled'} "
            f"(batch_size={batch_size}, cpus={cpu_count})"
        )

        return use_parallel


class InferenceOptimizer:
    """
    Unified inference optimization coordinator.

    Combines Metal cache management, memory awareness, and batch optimization
    for comprehensive performance tuning.
    """

    def __init__(self):
        """Initialize inference optimizer."""
        self.metal_cache = MetalCacheManager()
        self.memory_monitor = MemoryMonitor()
        self.memory_aware = MemoryAwareGenerator(self.memory_monitor)
        self.batch_optimizer = BatchOptimizer(self.memory_monitor)

    def prepare_for_generation(
        self,
        sampling_params: SamplingParams,
        prompt_tokens: int
    ) -> SamplingParams:
        """
        Prepare and optimize parameters for generation.

        Args:
            sampling_params: Original sampling parameters
            prompt_tokens: Number of tokens in prompt

        Returns:
            Optimized SamplingParams
        """
        # Clear Metal cache if needed
        self.metal_cache.optimize_for_generation(sampling_params.max_tokens)

        # Adjust for memory
        optimized_params = self.memory_aware.adjust_max_tokens(
            sampling_params, prompt_tokens
        )

        return optimized_params

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization status report.

        Returns:
            Dictionary with optimization status
        """
        mem_stats = self.memory_monitor.get_memory_stats()
        metal_stats = self.metal_cache.get_cache_stats()

        return {
            "memory": {
                "available_gb": mem_stats.available_gb,
                "used_gb": mem_stats.used_gb,
                "percent_used": mem_stats.percent_used,
            },
            "metal": metal_stats,
            "system": {
                "cpu_count": os.cpu_count(),
            }
        }
