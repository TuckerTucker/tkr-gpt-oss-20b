"""
Inference-specific exception classes.

This module defines exception classes for handling inference-related errors.
"""

from typing import Optional


class InferenceError(Exception):
    """Base exception for inference operations."""

    def __init__(self, message: str, details: Optional[str] = None):
        """
        Initialize InferenceError.

        Args:
            message: Primary error message
            details: Additional context or debugging information
        """
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class GenerationError(InferenceError):
    """Exception raised when text generation fails."""

    def __init__(
        self,
        message: str,
        prompt: Optional[str] = None,
        details: Optional[str] = None
    ):
        """
        Initialize GenerationError.

        Args:
            message: Primary error message
            prompt: The prompt that caused the error (truncated if long)
            details: Additional context or debugging information
        """
        self.prompt = prompt[:100] + "..." if prompt and len(prompt) > 100 else prompt
        super().__init__(message, details)

    def __str__(self) -> str:
        """Return formatted error message."""
        base_msg = super().__str__()
        if self.prompt:
            return f"{base_msg}\nPrompt: {self.prompt}"
        return base_msg


class ModelNotLoadedError(InferenceError):
    """Exception raised when attempting inference without a loaded model."""

    def __init__(self, message: str = "Model not loaded. Please load a model before inference."):
        """
        Initialize ModelNotLoadedError.

        Args:
            message: Error message
        """
        super().__init__(message)


class InvalidSamplingParamsError(InferenceError):
    """Exception raised when sampling parameters are invalid."""

    def __init__(self, param_name: str, param_value, reason: str):
        """
        Initialize InvalidSamplingParamsError.

        Args:
            param_name: Name of the invalid parameter
            param_value: Value that was invalid
            reason: Explanation of why it's invalid
        """
        self.param_name = param_name
        self.param_value = param_value
        self.reason = reason
        message = f"Invalid sampling parameter '{param_name}' = {param_value}: {reason}"
        super().__init__(message)


class GenerationCancelledError(InferenceError):
    """Exception raised when generation is cancelled by user."""

    def __init__(self, message: str = "Text generation was cancelled."):
        """
        Initialize GenerationCancelledError.

        Args:
            message: Error message
        """
        super().__init__(message)


class ContextLengthExceededError(InferenceError):
    """Exception raised when prompt exceeds model's context length."""

    def __init__(
        self,
        prompt_length: int,
        max_length: int,
        message: Optional[str] = None
    ):
        """
        Initialize ContextLengthExceededError.

        Args:
            prompt_length: Actual prompt length in tokens
            max_length: Maximum allowed context length
            message: Custom error message (optional)
        """
        self.prompt_length = prompt_length
        self.max_length = max_length
        if message is None:
            message = (
                f"Prompt length ({prompt_length} tokens) exceeds "
                f"maximum context length ({max_length} tokens)"
            )
        super().__init__(message)
