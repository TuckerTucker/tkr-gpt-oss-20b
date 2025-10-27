"""
MLX-based model loader for Apple Silicon.

Provides model loading and management using mlx-lm, optimized for
Apple Silicon with Metal acceleration.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""

    model_name: str  # Model identifier
    model_path: str  # HuggingFace repository path
    quantization: str  # Quantization level (4bit, 8bit, fp16)
    device: str  # Device ("mps" or "cpu")
    memory_usage_mb: int  # Actual memory usage
    context_length: int  # Maximum context window
    is_loaded: bool  # Whether model is currently loaded
    load_time_ms: int  # Time taken to load in milliseconds


class ModelLoader:
    """
    Load and manage MLX models on Apple Silicon.

    Responsibilities:
    - Model loading with mlx-lm
    - Device validation (Apple Silicon required)
    - Memory management and reporting
    - Error handling (OOM, missing models, device errors)
    - Model warmup and health checks
    """

    def __init__(self, config: Any):
        """
        Initialize loader with configuration.

        Args:
            config: ModelConfig with model settings

        Raises:
            ValueError: Invalid configuration
        """
        # Import here to avoid circular dependency
        # In real implementation, config would be typed as ModelConfig
        self.config = config

        # Validate config if it has validate method
        if hasattr(config, "validate"):
            config.validate()

        # State
        self._model = None
        self._tokenizer = None
        self._model_info: Optional[ModelInfo] = None
        self._is_loaded = False

        # Extract config values
        self.model_name = getattr(config, "model_name", "phi-3-mini")
        self.quantization = getattr(config, "quantization", "4bit")
        self.device = getattr(config, "device", "auto")
        self.max_model_len = getattr(config, "max_model_len", 4096)
        self.lazy_load = getattr(config, "lazy_load", True)
        self.warmup = getattr(config, "warmup", True)
        self.model_path_override = getattr(config, "model_path", None)

        logger.info(f"ModelLoader initialized for {self.model_name}")

    def load(self) -> ModelInfo:
        """
        Load model according to config.

        Returns:
            ModelInfo with metadata

        Raises:
            ModelLoadError: Failed to load model
            OutOfMemoryError: Insufficient memory
            DeviceError: Device not available
        """
        from .exceptions import ModelLoadError, OutOfMemoryError, DeviceError
        from .registry import get_model_spec
        from .memory import MemoryManager
        from ..devices import DeviceDetector

        logger.info(f"Loading model: {self.model_name}")
        start_time = time.time()

        try:
            # 1. Validate platform
            is_valid, message = DeviceDetector.validate_platform()
            if not is_valid:
                raise DeviceError("mps", message)

            # 2. Detect device
            detected_device = DeviceDetector.detect()
            if self.device == "auto":
                self.device = detected_device
            logger.info(f"Using device: {self.device}")

            # 3. Get model spec from registry (or use direct HuggingFace path)
            spec = get_model_spec(self.model_name)

            if spec is None:
                # Not in registry - try as direct HuggingFace path
                logger.info(f"Model not in registry, attempting direct HuggingFace path: {self.model_name}")
                model_path = self.model_path_override or self.model_name
                # Use reasonable defaults for unknown models
                context_length = self.max_model_len
            else:
                # Use registry spec
                model_path = self.model_path_override or spec["path"]
                context_length = spec["context_length"]

            # 4. Check memory availability
            can_fit, mem_message = MemoryManager.can_fit_model(
                self.model_name, self.quantization, self.device
            )
            logger.info(mem_message)

            if not can_fit:
                required = MemoryManager.estimate_memory_requirement(
                    self.model_name, self.quantization
                )
                available = MemoryManager.get_available_memory() or 0
                raise OutOfMemoryError(required, available, self.device)

            # 5. Load model with mlx-lm
            logger.info(f"Loading model from {model_path}...")

            try:
                import mlx_lm

                self._model, self._tokenizer = mlx_lm.load(model_path)
                logger.info("✓ Model loaded successfully")

            except ImportError:
                raise ModelLoadError(
                    self.model_name,
                    "mlx-lm not installed. Install with: pip install mlx-lm",
                )
            except Exception as e:
                raise ModelLoadError(
                    self.model_name,
                    f"Failed to load from HuggingFace: {str(e)}",
                )

            # 6. Record memory usage
            memory_usage = MemoryManager.get_model_memory_usage()

            # 7. Warmup if requested
            if self.warmup:
                logger.info("Warming up model with dummy inference...")
                self._warmup()

            # 8. Mark as loaded
            self._is_loaded = True

            # 9. Create model info
            load_time = int((time.time() - start_time) * 1000)  # ms

            self._model_info = ModelInfo(
                model_name=self.model_name,
                model_path=model_path,
                quantization=self.quantization,
                device=self.device,
                memory_usage_mb=memory_usage,
                context_length=context_length,
                is_loaded=True,
                load_time_ms=load_time,
            )

            logger.info(
                f"✓ Model loaded in {load_time}ms, using {memory_usage}MB"
            )

            return self._model_info

        except (ModelLoadError, OutOfMemoryError, DeviceError) as e:
            # Re-raise model-specific exceptions
            logger.error(f"Failed to load model: {e}")
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            logger.error(f"Unexpected error loading model: {e}", exc_info=True)
            from .exceptions import ModelLoadError

            raise ModelLoadError(self.model_name, f"Unexpected error: {str(e)}")

    def _warmup(self) -> None:
        """
        Warm up model by generating dummy tokens.

        This ensures the model is fully loaded into memory and Metal
        kernels are compiled.
        """
        try:
            import mlx_lm

            # Generate a few tokens with a simple prompt
            warmup_prompt = "Hello"
            mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=warmup_prompt,
                max_tokens=10,
                verbose=False,
            )
            logger.debug("Model warmup complete")

        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.

        Returns:
            True if model is loaded and ready for inference
        """
        return self._is_loaded and self._model is not None

    def unload(self) -> None:
        """
        Unload model and free memory.

        Releases model and tokenizer references, allowing Metal to
        reclaim memory.
        """
        if not self.is_loaded():
            logger.debug("Model not loaded, nothing to unload")
            return

        logger.info(f"Unloading model: {self.model_name}")

        # Release references
        self._model = None
        self._tokenizer = None
        self._is_loaded = False

        if self._model_info:
            self._model_info.is_loaded = False

        # Clear Metal cache
        from .memory import MemoryManager

        MemoryManager.clear_cache()

        logger.info("✓ Model unloaded")

    def get_model(self) -> tuple[Any, Any]:
        """
        Get the loaded model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        from .exceptions import ModelNotLoadedError

        if not self.is_loaded():
            raise ModelNotLoadedError("get_model")

        return self._model, self._tokenizer

    def get_info(self) -> ModelInfo:
        """
        Get metadata about loaded model.

        Returns:
            ModelInfo with current metadata

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        from .exceptions import ModelNotLoadedError

        if self._model_info is None:
            raise ModelNotLoadedError("get_info")

        # Update memory usage
        from .memory import MemoryManager

        current_memory = MemoryManager.get_model_memory_usage()
        self._model_info.memory_usage_mb = current_memory

        return self._model_info

    def health_check(self) -> bool:
        """
        Verify model is healthy and can generate.

        Returns:
            True if model responds correctly

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        from .exceptions import ModelNotLoadedError

        if not self.is_loaded():
            raise ModelNotLoadedError("health_check")

        try:
            import mlx_lm

            # Try a simple generation
            test_prompt = "Test"
            output = mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=test_prompt,
                max_tokens=5,
                verbose=False,
            )

            # Check that we got some output
            is_healthy = output is not None and len(output) > 0

            if is_healthy:
                logger.debug("Health check passed")
            else:
                logger.warning("Health check failed: no output generated")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def switch_model(self, new_config: Any) -> ModelInfo:
        """
        Switch to a different model.

        Args:
            new_config: Configuration for new model

        Returns:
            ModelInfo for newly loaded model

        Raises:
            ModelLoadError: Failed to load new model
        """
        logger.info(f"Switching model from {self.model_name} to {new_config.model_name}")

        # Unload current model
        if self.is_loaded():
            self.unload()

        # Update config
        self.config = new_config
        self.model_name = getattr(new_config, "model_name", "phi-3-mini")
        self.quantization = getattr(new_config, "quantization", "4bit")
        self.device = getattr(new_config, "device", "auto")
        self.max_model_len = getattr(new_config, "max_model_len", 4096)
        self.warmup = getattr(new_config, "warmup", True)
        self.model_path_override = getattr(new_config, "model_path", None)

        # Load new model
        return self.load()
