"""Model configuration for MLX-optimized inference.

This module provides ModelConfig dataclass for managing model loading,
quantization, device selection, and memory management settings.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Configuration for model loading and management.

    Attributes:
        model_name: Model variant to load (e.g., "gpt-j-20b")
        model_path: Optional override for HuggingFace model path
        quantization: Quantization level for memory optimization
        device: Compute device selection
        tensor_parallel_size: Number of GPUs for model parallelism
        gpu_memory_utilization: Fraction of GPU memory to use (0.5-1.0)
        max_model_len: Maximum context window size in tokens
        lazy_load: Defer model loading until first use
        warmup: Generate dummy tokens after loading
        trust_remote_code: Allow loading custom model code
    """

    # Model selection
    model_name: str = "gpt-j-20b"
    model_path: Optional[str] = None

    # Quantization
    quantization: Literal["int4", "int8", "fp16", "none"] = "int4"

    # Device
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    tensor_parallel_size: int = 1

    # Memory management
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096

    # Loading behavior
    lazy_load: bool = True
    warmup: bool = True
    trust_remote_code: bool = True

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate GPU memory utilization
        if self.gpu_memory_utilization < 0.5 or self.gpu_memory_utilization > 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be 0.5-1.0, got {self.gpu_memory_utilization}"
            )

        # Validate tensor parallel size
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

        # Validate max model length
        if self.max_model_len < 512 or self.max_model_len > 32768:
            raise ValueError(
                f"max_model_len must be 512-32768, got {self.max_model_len}"
            )

        # Validate quantization mode
        valid_quant = {"int4", "int8", "fp16", "none"}
        if self.quantization not in valid_quant:
            raise ValueError(
                f"quantization must be one of {valid_quant}, got {self.quantization}"
            )

        # Validate device
        valid_devices = {"cuda", "mps", "cpu", "auto"}
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got {self.device}"
            )

        # Validate model name
        if not self.model_name:
            raise ValueError("model_name cannot be empty")

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Load configuration from environment variables.

        Environment variables:
            MODEL_NAME: Model variant (default: gpt-j-20b)
            MODEL_PATH: Optional HuggingFace model path override
            QUANTIZATION: Quantization mode (default: int4)
            DEVICE: Compute device (default: auto)
            TENSOR_PARALLEL_SIZE: Number of GPUs (default: 1)
            GPU_MEMORY_UTIL: GPU memory fraction (default: 0.90)
            MAX_MODEL_LEN: Context window size (default: 4096)
            LAZY_LOAD: Defer model loading (default: true)
            WARMUP: Run warmup generation (default: true)
            TRUST_REMOTE_CODE: Allow custom model code (default: true)

        Returns:
            ModelConfig instance with values from environment.
        """
        # Helper to parse boolean env vars
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ("true", "1", "yes", "on")

        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-j-20b"),
            model_path=os.getenv("MODEL_PATH"),
            quantization=os.getenv("QUANTIZATION", "int4"),  # type: ignore
            device=os.getenv("DEVICE", "auto"),  # type: ignore
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTIL", "0.90")),
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            lazy_load=get_bool("LAZY_LOAD", True),
            warmup=get_bool("WARMUP", True),
            trust_remote_code=get_bool("TRUST_REMOTE_CODE", True),
        )
