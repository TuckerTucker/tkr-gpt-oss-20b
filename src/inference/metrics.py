"""
Metrics tracking for inference operations.

This module provides utilities for tracking and reporting inference performance metrics
such as latency, throughput, and token statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """
    Metrics for a single generation operation.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        tokens_generated: Number of tokens generated
        latency_ms: Total generation time in milliseconds
        time_to_first_token_ms: Time to generate first token (streaming)
        tokens_per_second: Generation throughput
        finish_reason: Reason generation stopped ("stop", "length", "error")
        timestamp: Unix timestamp when generation started
    """

    prompt_tokens: int
    tokens_generated: int
    latency_ms: int
    time_to_first_token_ms: Optional[int] = None
    tokens_per_second: float = 0.0
    finish_reason: str = "stop"
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.latency_ms > 0 and self.tokens_generated > 0:
            self.tokens_per_second = (self.tokens_generated / self.latency_ms) * 1000

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "tokens_generated": self.tokens_generated,
            "latency_ms": self.latency_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "tokens_per_second": self.tokens_per_second,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp,
        }


class MetricsTracker:
    """
    Track and aggregate inference metrics across multiple generations.

    This class maintains statistics about inference performance and can generate
    summary reports.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.generations: List[GenerationMetrics] = []
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None

    def start_generation(self) -> None:
        """Mark the start of a generation operation."""
        self._start_time = time.time()
        self._first_token_time = None

    def mark_first_token(self) -> None:
        """Mark when the first token is generated (for streaming)."""
        if self._start_time is not None and self._first_token_time is None:
            self._first_token_time = time.time()

    def end_generation(
        self,
        prompt_tokens: int,
        tokens_generated: int,
        finish_reason: str = "stop"
    ) -> GenerationMetrics:
        """
        Mark the end of a generation operation and record metrics.

        Args:
            prompt_tokens: Number of tokens in the prompt
            tokens_generated: Number of tokens generated
            finish_reason: Reason generation stopped

        Returns:
            GenerationMetrics for this operation
        """
        if self._start_time is None:
            logger.warning("end_generation called without start_generation")
            self._start_time = time.time()

        end_time = time.time()
        latency_ms = int((end_time - self._start_time) * 1000)

        time_to_first_token_ms = None
        if self._first_token_time is not None:
            time_to_first_token_ms = int((self._first_token_time - self._start_time) * 1000)

        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            tokens_generated=tokens_generated,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            finish_reason=finish_reason,
        )

        self.generations.append(metrics)
        self._start_time = None
        self._first_token_time = None

        return metrics

    def get_summary(self) -> Dict:
        """
        Get summary statistics across all tracked generations.

        Returns:
            Dictionary containing aggregate metrics
        """
        if not self.generations:
            return {
                "total_generations": 0,
                "total_tokens_generated": 0,
                "avg_latency_ms": 0.0,
                "avg_tokens_per_second": 0.0,
                "avg_time_to_first_token_ms": 0.0,
            }

        total_tokens = sum(g.tokens_generated for g in self.generations)
        total_latency = sum(g.latency_ms for g in self.generations)
        avg_latency = total_latency / len(self.generations)

        # Calculate average tokens per second
        tps_values = [g.tokens_per_second for g in self.generations if g.tokens_per_second > 0]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0

        # Calculate average time to first token (for streaming operations)
        ttft_values = [
            g.time_to_first_token_ms
            for g in self.generations
            if g.time_to_first_token_ms is not None
        ]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0

        return {
            "total_generations": len(self.generations),
            "total_tokens_generated": total_tokens,
            "avg_latency_ms": avg_latency,
            "avg_tokens_per_second": avg_tps,
            "avg_time_to_first_token_ms": avg_ttft,
            "min_latency_ms": min(g.latency_ms for g in self.generations),
            "max_latency_ms": max(g.latency_ms for g in self.generations),
        }

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self.generations.clear()
        self._start_time = None
        self._first_token_time = None

    def get_recent_metrics(self, n: int = 10) -> List[GenerationMetrics]:
        """
        Get metrics for the most recent n generations.

        Args:
            n: Number of recent generations to return

        Returns:
            List of GenerationMetrics
        """
        return self.generations[-n:]

    def log_summary(self) -> None:
        """Log a summary of tracked metrics."""
        summary = self.get_summary()
        logger.info(
            f"Inference Metrics Summary: "
            f"{summary['total_generations']} generations, "
            f"{summary['total_tokens_generated']} tokens, "
            f"{summary['avg_tokens_per_second']:.2f} tokens/sec avg"
        )
