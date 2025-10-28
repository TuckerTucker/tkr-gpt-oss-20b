"""Inference configuration for text generation.

This module provides InferenceConfig dataclass for managing sampling parameters,
streaming behavior, and stop sequences during text generation.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReasoningLevel(Enum):
    """Reasoning effort level for GPT-OSS models.

    Based on GPT-OSS-20B best practices:
    - LOW: Fast responses, minimal chain-of-thought (extraction, simple queries)
    - MEDIUM: Balanced quality/speed (chat, summarization) [DEFAULT]
    - HIGH: Maximum reasoning (research, complex analysis, open-ended tasks)
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class InferenceConfig:
    """Configuration for text generation.

    Attributes:
        temperature: Sampling temperature (0.0-2.0), higher = more random
        top_p: Nucleus sampling threshold (0.0-1.0)
        top_k: Top-k sampling limit, -1 to disable
        repetition_penalty: Penalty for token repetition (1.0 = no penalty)
        max_tokens: Maximum number of tokens to generate
        use_harmony_format: Enable Harmony multi-channel format parsing
        reasoning_level: Model reasoning effort (LOW/MEDIUM/HIGH)
        capture_reasoning: Store analysis channel in results
        show_reasoning: Display reasoning traces in CLI
        knowledge_cutoff: Model knowledge cutoff date (e.g., "2024-06")
        current_date: Current date for context (auto-detected if None)
        streaming: Enable token-by-token streaming output
        stop_sequences: List of sequences that stop generation
    """

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_tokens: int = 512

    # Harmony format settings
    use_harmony_format: bool = True
    reasoning_level: ReasoningLevel = ReasoningLevel.MEDIUM
    capture_reasoning: bool = False
    show_reasoning: bool = False
    knowledge_cutoff: str = "2024-06"
    current_date: Optional[str] = None  # Auto-detect if None

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

        # Validate reasoning_level
        if not isinstance(self.reasoning_level, ReasoningLevel):
            raise ValueError(
                f"reasoning_level must be ReasoningLevel enum, got {type(self.reasoning_level)}"
            )

        # Validate knowledge_cutoff
        if not self.knowledge_cutoff:
            raise ValueError("knowledge_cutoff cannot be empty")

        # Validate current_date format if provided
        if self.current_date:
            if len(self.current_date) != 10:
                raise ValueError(
                    f"current_date must be YYYY-MM-DD format, got {self.current_date}"
                )
            # Basic format check: should be YYYY-MM-DD
            if not (self.current_date[4] == '-' and self.current_date[7] == '-'):
                raise ValueError(
                    f"current_date must be YYYY-MM-DD format, got {self.current_date}"
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
            USE_HARMONY_FORMAT: Enable Harmony format parsing (default: true)
            REASONING_LEVEL: Reasoning effort - low/medium/high (default: medium)
            CAPTURE_REASONING: Store analysis channel (default: false)
            SHOW_REASONING: Display reasoning traces (default: false)
            KNOWLEDGE_CUTOFF: Model knowledge cutoff date (default: 2024-06)
            CURRENT_DATE: Current date for context (default: auto-detect)
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

        # Harmony settings
        use_harmony_format = get_bool("USE_HARMONY_FORMAT", True)

        reasoning_level_str = os.getenv("REASONING_LEVEL", "medium").lower()
        reasoning_level = ReasoningLevel.MEDIUM
        if reasoning_level_str == "low":
            reasoning_level = ReasoningLevel.LOW
        elif reasoning_level_str == "high":
            reasoning_level = ReasoningLevel.HIGH

        capture_reasoning = get_bool("CAPTURE_REASONING", False)
        show_reasoning = get_bool("SHOW_REASONING", False)

        # Load knowledge cutoff
        knowledge_cutoff = os.getenv("KNOWLEDGE_CUTOFF", "2024-06")

        # Load current date (with auto-detection)
        current_date = os.getenv("CURRENT_DATE")
        if current_date is None or current_date.strip() == "":
            from datetime import datetime
            current_date = datetime.now().strftime('%Y-%m-%d')

        return cls(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_p=float(os.getenv("TOP_P", "0.9")),
            top_k=int(os.getenv("TOP_K", "50")),
            repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "512")),
            use_harmony_format=use_harmony_format,
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
            show_reasoning=show_reasoning,
            knowledge_cutoff=knowledge_cutoff,
            current_date=current_date,
            streaming=get_bool("STREAMING", True),
            stop_sequences=stop_sequences,
        )
