"""
Exception hierarchy for model loading and management.

Provides specific exception classes for different failure modes in model
operations, enabling targeted error handling and user-friendly messages.
"""


class ModelError(Exception):
    """
    Base exception for model operations.

    All model-related exceptions inherit from this base class,
    allowing consumers to catch all model errors with a single handler.
    """

    pass


class ModelLoadError(ModelError):
    """
    Failed to load model.

    Raised when model loading fails for reasons such as:
    - Model not found on HuggingFace
    - Invalid model format
    - Network issues during download
    - Configuration errors
    """

    def __init__(self, model_name: str, reason: str):
        """
        Initialize ModelLoadError.

        Args:
            model_name: Name of the model that failed to load
            reason: Detailed reason for the failure
        """
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to load model '{model_name}': {reason}")


class OutOfMemoryError(ModelError):
    """
    Insufficient memory for model.

    Raised when there is not enough VRAM/RAM to load or run the model.
    Provides information about memory requirements vs availability to help
    users choose appropriate quantization or model size.
    """

    def __init__(self, required_mb: int, available_mb: int, device: str = "Metal"):
        """
        Initialize OutOfMemoryError.

        Args:
            required_mb: Required memory in megabytes
            available_mb: Available memory in megabytes
            device: Device type (e.g., "Metal", "CPU")
        """
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.device = device
        super().__init__(
            f"Out of {device} memory: need ~{required_mb}MB, have {available_mb}MB available. "
            f"Try using a more aggressive quantization or a smaller model."
        )


class DeviceError(ModelError):
    """
    Requested device not available.

    Raised when the requested compute device (e.g., Apple Silicon Metal)
    is not available on the current platform.
    """

    def __init__(self, device: str, reason: str):
        """
        Initialize DeviceError.

        Args:
            device: Device that was requested
            reason: Why the device is not available
        """
        self.device = device
        self.reason = reason
        super().__init__(f"Device '{device}' not available: {reason}")


class ModelNotFoundError(ModelError):
    """
    Model not found in registry or HuggingFace.

    Raised when the requested model cannot be found locally or remotely.
    """

    def __init__(self, model_name: str, searched_locations: list[str] = None):
        """
        Initialize ModelNotFoundError.

        Args:
            model_name: Name of the model that was not found
            searched_locations: List of locations that were searched (optional)
        """
        self.model_name = model_name
        self.searched_locations = searched_locations or []

        locations_str = ""
        if self.searched_locations:
            locations_str = f" Searched: {', '.join(self.searched_locations)}"

        super().__init__(
            f"Model '{model_name}' not found in registry or HuggingFace.{locations_str}"
        )


class ModelNotLoadedError(ModelError):
    """
    Attempted operation on unloaded model.

    Raised when trying to use a model that hasn't been loaded yet,
    or has been unloaded.
    """

    def __init__(self, operation: str):
        """
        Initialize ModelNotLoadedError.

        Args:
            operation: The operation that was attempted
        """
        self.operation = operation
        super().__init__(
            f"Cannot perform '{operation}': model is not loaded. "
            f"Call load() first."
        )


class QuantizationError(ModelError):
    """
    Invalid or unsupported quantization configuration.

    Raised when the requested quantization is not supported for the model
    or is invalid.
    """

    def __init__(self, quantization: str, model_name: str, supported: list[str] = None):
        """
        Initialize QuantizationError.

        Args:
            quantization: The quantization that was requested
            model_name: Name of the model
            supported: List of supported quantizations (optional)
        """
        self.quantization = quantization
        self.model_name = model_name
        self.supported = supported or []

        supported_str = ""
        if self.supported:
            supported_str = f" Supported quantizations: {', '.join(self.supported)}"

        super().__init__(
            f"Quantization '{quantization}' not supported for model '{model_name}'.{supported_str}"
        )
