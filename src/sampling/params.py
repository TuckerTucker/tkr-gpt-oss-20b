"""
Sampling parameters for text generation.

This module provides validation and management of sampling parameters
used during text generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    """
    Parameters controlling text generation sampling behavior.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0). Cumulative probability cutoff.
        top_k: Top-k sampling. Only consider top k tokens. 0 = disabled.
        max_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
        presence_penalty: Penalty for tokens that have appeared (0.0 = no penalty).
        frequency_penalty: Penalty based on token frequency (0.0 = no penalty).
        stop_sequences: List of sequences that stop generation when encountered.
        min_tokens: Minimum number of tokens to generate before considering stop.
        seed: Random seed for reproducible generation. None = random.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    min_tokens: int = 0
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all sampling parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Temperature validation
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        # Top-p validation
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be between 0.0 and 1.0, got {self.top_p}"
            )

        # Top-k validation
        if self.top_k < 0:
            raise ValueError(
                f"top_k must be non-negative, got {self.top_k}"
            )

        # Max tokens validation
        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        # Min tokens validation
        if self.min_tokens < 0:
            raise ValueError(
                f"min_tokens must be non-negative, got {self.min_tokens}"
            )

        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens ({self.min_tokens}) cannot exceed max_tokens ({self.max_tokens})"
            )

        # Repetition penalty validation
        if self.repetition_penalty < 0.0:
            raise ValueError(
                f"repetition_penalty must be non-negative, got {self.repetition_penalty}"
            )

        # Presence penalty validation
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, got {self.presence_penalty}"
            )

        # Frequency penalty validation
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be between -2.0 and 2.0, got {self.frequency_penalty}"
            )

        # Stop sequences validation
        if self.stop_sequences is not None:
            if not isinstance(self.stop_sequences, list):
                raise ValueError(
                    f"stop_sequences must be a list, got {type(self.stop_sequences)}"
                )
            for seq in self.stop_sequences:
                if not isinstance(seq, str):
                    raise ValueError(
                        f"stop_sequences must contain strings, got {type(seq)}"
                    )

        # Seed validation
        if self.seed is not None and self.seed < 0:
            raise ValueError(
                f"seed must be non-negative, got {self.seed}"
            )

    def to_dict(self) -> dict:
        """
        Convert parameters to dictionary.

        Returns:
            Dictionary representation of parameters
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_sequences": self.stop_sequences,
            "min_tokens": self.min_tokens,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, params_dict: dict) -> "SamplingParams":
        """
        Create SamplingParams from dictionary.

        Args:
            params_dict: Dictionary of parameters

        Returns:
            SamplingParams instance

        Raises:
            ValueError: If parameters are invalid
        """
        # Filter to only known parameters
        valid_params = {
            k: v for k, v in params_dict.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**valid_params)

    def copy(self, **overrides) -> "SamplingParams":
        """
        Create a copy with optional parameter overrides.

        Args:
            **overrides: Parameters to override

        Returns:
            New SamplingParams instance with overrides applied
        """
        params_dict = self.to_dict()
        params_dict.update(overrides)
        return self.from_dict(params_dict)

    def __repr__(self) -> str:
        """Return string representation of parameters."""
        return (
            f"SamplingParams("
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"max_tokens={self.max_tokens}, "
            f"repetition_penalty={self.repetition_penalty})"
        )


# Preset sampling configurations
DEFAULT_SAMPLING = SamplingParams()

CREATIVE_SAMPLING = SamplingParams(
    temperature=1.2,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
)

PRECISE_SAMPLING = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    top_k=20,
    repetition_penalty=1.05,
)

DETERMINISTIC_SAMPLING = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    seed=42,
)
