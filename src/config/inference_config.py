"""Inference configuration for text generation.

This module provides InferenceConfig dataclass for managing sampling parameters,
streaming behavior, and stop sequences during text generation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for text generation.

    Attributes:
        temperature: Sampling temperature (0.0-2.0), higher = more random
        top_p: Nucleus sampling threshold (0.0-1.0)
        top_k: Top-k sampling limit, -1 to disable
        repetition_penalty: Penalty for token repetition (1.0 = no penalty)
        max_tokens: Maximum number of tokens to generate
        streaming: Enable token-by-token streaming output
        stop_sequences: List of sequences that stop generation
    """

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_tokens: int = 512

    # Streaming
    streaming: bool = True

    # Stop sequences
    stop_sequences: Optional[list[str]] = None

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate temperature
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(
                f"temperature must be 0.0-2.0, got {self.temperature}"
            )

        # Validate top_p
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError(
                f"top_p must be 0.0-1.0, got {self.top_p}"
            )

        # Validate max_tokens
        if self.max_tokens < 1 or self.max_tokens > 4096:
            raise ValueError(
                f"max_tokens must be 1-4096, got {self.max_tokens}"
            )

        # Validate repetition penalty
        if self.repetition_penalty < 0.0:
            raise ValueError(
                f"repetition_penalty must be >= 0.0, got {self.repetition_penalty}"
            )

        # Validate top_k
        if self.top_k < -1:
            raise ValueError(
                f"top_k must be >= -1, got {self.top_k}"
            )

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Load configuration from environment variables.

        Environment variables:
            TEMPERATURE: Sampling temperature (default: 0.7)
            TOP_P: Nucleus sampling threshold (default: 0.9)
            TOP_K: Top-k sampling limit (default: 50)
            REPETITION_PENALTY: Repetition penalty factor (default: 1.0)
            MAX_TOKENS: Maximum output tokens (default: 512)
            STREAMING: Enable streaming output (default: true)
            STOP_SEQUENCES: Comma-separated stop sequences (optional)

        Returns:
            InferenceConfig instance with values from environment.
        """
        # Helper to parse boolean env vars
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ("true", "1", "yes", "on")

        # Parse stop sequences from comma-separated string
        stop_sequences = None
        stop_str = os.getenv("STOP_SEQUENCES")
        if stop_str:
            stop_sequences = [s.strip() for s in stop_str.split(",") if s.strip()]

        return cls(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            top_k=int(os.getenv("TOP_K", "50")),
            repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            streaming=get_bool("STREAMING", True),
            stop_sequences=stop_sequences,
        )
