"""
Error recovery strategies for inference operations.

This module provides retry strategies, graceful degradation, and error
classification for handling transient and permanent failures during inference.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Any, Type
from enum import Enum
import time
import logging

from src.inference.exceptions import (
    InferenceError,
    GenerationError,
    ModelNotLoadedError,
    ContextLengthExceededError,
)
from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types."""
    RECOVERABLE = "recoverable"  # Transient errors that can be retried
    DEGRADABLE = "degradable"  # Errors that can be handled with degradation
    FATAL = "fatal"  # Permanent errors that cannot be recovered


@dataclass
class ErrorClassification:
    """
    Classification result for an error.

    Attributes:
        error_type: Type of error (recoverable, degradable, fatal)
        can_retry: Whether the operation can be retried
        can_degrade: Whether graceful degradation is possible
        suggested_action: Recommended action to take
        details: Additional context about the classification
    """
    error_type: ErrorType
    can_retry: bool
    can_degrade: bool
    suggested_action: str
    details: Optional[str] = None


class ErrorClassifier:
    """
    Classify errors to determine recovery strategy.

    Analyzes exceptions to determine whether they're recoverable,
    require degradation, or are fatal.
    """

    @staticmethod
    def classify(error: Exception) -> ErrorClassification:
        """
        Classify an error and determine recovery strategy.

        Args:
            error: The exception to classify

        Returns:
            ErrorClassification with recovery recommendations
        """
        error_str = str(error).lower()

        # Fatal errors - cannot recover
        if isinstance(error, ModelNotLoadedError):
            return ErrorClassification(
                error_type=ErrorType.FATAL,
                can_retry=False,
                can_degrade=False,
                suggested_action="Load model before inference",
                details="Model must be loaded"
            )

        # Degradable errors - can reduce requirements
        if isinstance(error, ContextLengthExceededError):
            return ErrorClassification(
                error_type=ErrorType.DEGRADABLE,
                can_retry=True,
                can_degrade=True,
                suggested_action="Reduce max_tokens or truncate prompt",
                details="Context length exceeded"
            )

        if "memory" in error_str or "oom" in error_str or "out of memory" in error_str:
            return ErrorClassification(
                error_type=ErrorType.DEGRADABLE,
                can_retry=True,
                can_degrade=True,
                suggested_action="Reduce max_tokens or batch size",
                details="Out of memory"
            )

        # Recoverable errors - transient failures
        if any(keyword in error_str for keyword in [
            "timeout", "connection", "temporary", "unavailable", "busy"
        ]):
            return ErrorClassification(
                error_type=ErrorType.RECOVERABLE,
                can_retry=True,
                can_degrade=False,
                suggested_action="Retry with exponential backoff",
                details="Transient network or resource issue"
            )

        # Default to recoverable with caution
        return ErrorClassification(
            error_type=ErrorType.RECOVERABLE,
            can_retry=True,
            can_degrade=False,
            suggested_action="Retry once, then fail",
            details="Unknown error type"
        )


@dataclass
class RetryConfig:
    """
    Configuration for retry strategy.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff (2.0 = double each time)
        jitter: Whether to add random jitter to delays
        retry_on: Exception types to retry on (None = retry all)
    """
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: Optional[tuple[Type[Exception], ...]] = None


class RetryStrategy:
    """
    Retry strategy with exponential backoff for transient failures.

    Provides intelligent retry logic with configurable backoff and
    error classification.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry strategy.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
        self.classifier = ErrorClassifier()

    def execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from successful function execution

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        attempt = 0

        while attempt <= self.config.max_retries:
            try:
                # First attempt or retry
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.config.max_retries}")

                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry successful on attempt {attempt}")
                return result

            except Exception as e:
                last_error = e
                attempt += 1

                # Check if we should retry this error
                if self.config.retry_on and not isinstance(e, self.config.retry_on):
                    logger.debug(f"Error type {type(e)} not in retry_on list")
                    raise

                # Classify the error
                classification = self.classifier.classify(e)

                # Don't retry fatal errors
                if classification.error_type == ErrorType.FATAL:
                    logger.error(f"Fatal error, cannot retry: {e}")
                    raise

                # Check if we can retry
                if not classification.can_retry:
                    logger.error(f"Error not recoverable: {e}")
                    raise

                # Check if we've exhausted retries
                if attempt > self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) exceeded: {e}"
                    )
                    raise

                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Error on attempt {attempt}: {e}. "
                    f"Retrying in {delay:.2f}s... ({classification.suggested_action})"
                )
                time.sleep(delay)

        # Should never reach here, but handle gracefully
        if last_error:
            raise last_error
        raise InferenceError("Retry failed with no error recorded")

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        import random

        # Exponential backoff: initial_delay * (base ^ (attempt - 1))
        delay = self.config.initial_delay * (
            self.config.exponential_base ** (attempt - 1)
        )

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            # Add random jitter of ±25%
            jitter_factor = 1.0 + (random.random() - 0.5) * 0.5
            delay *= jitter_factor

        return delay


class GracefulDegradation:
    """
    Graceful degradation strategies for handling resource constraints.

    Automatically adjusts generation parameters when encountering
    memory or context length errors.
    """

    @staticmethod
    def reduce_max_tokens(
        sampling_params: SamplingParams,
        reduction_factor: float = 0.5
    ) -> SamplingParams:
        """
        Reduce max_tokens to handle memory constraints.

        Args:
            sampling_params: Original sampling parameters
            reduction_factor: Factor to reduce by (0.5 = reduce to 50%)

        Returns:
            New SamplingParams with reduced max_tokens
        """
        new_max_tokens = max(1, int(sampling_params.max_tokens * reduction_factor))
        logger.info(
            f"Reducing max_tokens from {sampling_params.max_tokens} "
            f"to {new_max_tokens}"
        )
        return sampling_params.copy(max_tokens=new_max_tokens)

    @staticmethod
    def truncate_prompt(
        prompt: str,
        max_length: int,
        truncation_strategy: str = "end"
    ) -> str:
        """
        Truncate prompt to fit within context length.

        Args:
            prompt: Original prompt text
            max_length: Maximum character length
            truncation_strategy: Where to truncate ("start", "end", "middle")

        Returns:
            Truncated prompt
        """
        if len(prompt) <= max_length:
            return prompt

        logger.warning(
            f"Truncating prompt from {len(prompt)} to {max_length} characters"
        )

        if truncation_strategy == "start":
            # Keep end of prompt
            return "..." + prompt[-(max_length - 3):]
        elif truncation_strategy == "middle":
            # Keep start and end, remove middle
            half = (max_length - 3) // 2
            return prompt[:half] + "..." + prompt[-half:]
        else:  # "end"
            # Keep start of prompt (default)
            return prompt[:max_length - 3] + "..."

    @staticmethod
    def create_degraded_params(
        original_params: SamplingParams,
        error: Exception
    ) -> Optional[SamplingParams]:
        """
        Create degraded parameters based on error type.

        Args:
            original_params: Original sampling parameters
            error: The error that occurred

        Returns:
            Degraded SamplingParams, or None if degradation not possible
        """
        classifier = ErrorClassifier()
        classification = classifier.classify(error)

        if not classification.can_degrade:
            return None

        error_str = str(error).lower()

        # Handle memory errors
        if "memory" in error_str or "oom" in error_str:
            # Reduce max_tokens significantly for memory issues
            return GracefulDegradation.reduce_max_tokens(
                original_params,
                reduction_factor=0.5
            )

        # Handle context length errors
        if isinstance(error, ContextLengthExceededError) or "context" in error_str:
            # Reduce max_tokens to leave more room for prompt
            return GracefulDegradation.reduce_max_tokens(
                original_params,
                reduction_factor=0.7
            )

        return None


class ResilientInference:
    """
    Wrapper providing resilient inference with retry and degradation.

    Combines retry strategy and graceful degradation for robust inference.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        enable_degradation: bool = True
    ):
        """
        Initialize resilient inference wrapper.

        Args:
            retry_config: Configuration for retry strategy
            enable_degradation: Whether to enable graceful degradation
        """
        self.retry_strategy = RetryStrategy(retry_config)
        self.enable_degradation = enable_degradation
        self.classifier = ErrorClassifier()

    def generate_with_recovery(
        self,
        engine,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> Any:
        """
        Generate with automatic recovery and degradation.

        Args:
            engine: InferenceEngine instance
            prompt: Input prompt
            sampling_params: Sampling parameters

        Returns:
            GenerationResult from successful generation

        Raises:
            Exception: If all recovery attempts fail
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        current_params = sampling_params
        current_prompt = prompt

        def _try_generate():
            return engine.generate(current_prompt, current_params)

        try:
            # First attempt with retry
            return self.retry_strategy.execute(_try_generate)

        except Exception as e:
            # Check if we can degrade
            if not self.enable_degradation:
                raise

            classification = self.classifier.classify(e)
            if not classification.can_degrade:
                raise

            logger.info(f"Attempting graceful degradation: {classification.suggested_action}")

            # Try with degraded parameters
            degraded_params = GracefulDegradation.create_degraded_params(
                current_params, e
            )
            if degraded_params is None:
                raise

            current_params = degraded_params

            # Retry with degraded parameters
            try:
                result = self.retry_strategy.execute(_try_generate)
                logger.info("Generation successful with degraded parameters")
                return result
            except Exception as e2:
                logger.error(f"Generation failed even with degradation: {e2}")
                raise
