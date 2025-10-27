"""
Integration tests for Config → ModelLoader workflow.

Tests the integration between configuration classes and model loading,
ensuring config values properly flow through to model initialization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from src.config.model_config import ModelConfig
from src.models.loader import ModelLoader, ModelInfo
from src.models.exceptions import (
    ModelLoadError,
    OutOfMemoryError,
    DeviceError,
)

logger = logging.getLogger(__name__)


class TestConfigToModelIntegration:
    """Test integration between ModelConfig and ModelLoader."""

    def test_model_loader_accepts_valid_config(self, mock_model_config):
        """Test that ModelLoader accepts and uses valid configuration."""
        # Create loader with config
        loader = ModelLoader(config=mock_model_config)

        # Verify config values were absorbed
        assert loader.model_name == mock_model_config.model_name
        assert loader.quantization == mock_model_config.quantization
        assert loader.device == mock_model_config.device
        assert loader.max_model_len == mock_model_config.max_model_len

    def test_model_loader_validates_config(self):
        """Test that ModelLoader validates configuration on init."""
        # Create invalid config (mock won't validate)
        invalid_config = Mock()
        invalid_config.validate = Mock(side_effect=ValueError("Invalid GPU memory"))
        invalid_config.model_name = "test"
        invalid_config.quantization = "int4"
        invalid_config.device = "auto"
        invalid_config.max_model_len = 4096
        invalid_config.lazy_load = True
        invalid_config.warmup = True
        invalid_config.model_path = None

        # Should raise during initialization
        with pytest.raises(ValueError, match="Invalid GPU memory"):
            ModelLoader(config=invalid_config)

    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    @patch('mlx_lm.load')
    def test_config_values_propagate_to_model_load(
        self,
        mock_mlx_load,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_model_config,
    ):
        """Test that config values properly flow through to model loading."""
        # Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Platform valid")
        mock_device_detector.detect.return_value = "mps"

        mock_get_model_spec.return_value = {
            "path": "mlx-community/gpt-j-20b",
            "context_length": 4096,
        }

        mock_memory_manager.can_fit_model.return_value = (True, "Model fits")
        mock_memory_manager.get_model_memory_usage.return_value = 7000

        # Mock mlx_lm.load to return model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        # Create loader and load model
        loader = ModelLoader(config=mock_model_config)
        model_info = loader.load()

        # Verify config values were used
        assert model_info.model_name == mock_model_config.model_name
        assert model_info.quantization == mock_model_config.quantization
        assert model_info.context_length == mock_model_config.max_model_len

    def test_config_device_auto_resolution(self):
        """Test that device='auto' is properly resolved."""
        # Create config with auto device
        config = ModelConfig(
            model_name="test-model",
            device="auto",
            quantization="int4",
        )

        # Verify config is valid
        config.validate()

        # Create loader (device should be 'auto' initially)
        loader = ModelLoader(config=config)
        assert loader.device == "auto"

    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_config_oom_error_handling(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
    ):
        """Test that OOM errors are properly raised when config demands too much memory."""
        # Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Platform valid")
        mock_device_detector.detect.return_value = "mps"

        mock_get_model_spec.return_value = {
            "path": "mlx-community/gpt-j-20b",
            "context_length": 4096,
        }

        # Simulate insufficient memory
        mock_memory_manager.can_fit_model.return_value = (False, "Insufficient memory")
        mock_memory_manager.estimate_memory_requirement.return_value = 16000
        mock_memory_manager.get_available_memory.return_value = 8000

        # Create config for large model
        config = ModelConfig(
            model_name="gpt-j-20b",
            quantization="fp16",  # Less aggressive quantization = more memory
            device="auto",
        )

        loader = ModelLoader(config=config)

        # Should raise OOM error
        with pytest.raises(OutOfMemoryError):
            loader.load()

    def test_config_from_env_to_loader(self, monkeypatch):
        """Test that config can be loaded from env and used with ModelLoader."""
        # Set environment variables
        monkeypatch.setenv("MODEL_NAME", "phi-3-mini")
        monkeypatch.setenv("QUANTIZATION", "int8")
        monkeypatch.setenv("DEVICE", "mps")
        monkeypatch.setenv("MAX_MODEL_LEN", "2048")
        monkeypatch.setenv("WARMUP", "false")

        # Load config from env
        config = ModelConfig.from_env()

        # Verify values
        assert config.model_name == "phi-3-mini"
        assert config.quantization == "int8"
        assert config.device == "mps"
        assert config.max_model_len == 2048
        assert config.warmup is False

        # Create loader with env-based config
        loader = ModelLoader(config=config)

        # Verify values propagated
        assert loader.model_name == "phi-3-mini"
        assert loader.quantization == "int8"
        assert loader.device == "mps"
        assert loader.max_model_len == 2048
        assert loader.warmup is False


class TestModelConfigValidationIntegration:
    """Test that ModelConfig validation works correctly with ModelLoader."""

    def test_invalid_quantization_rejected_before_load(self):
        """Test that invalid quantization is caught at config level."""
        with pytest.raises(ValueError, match="quantization"):
            config = ModelConfig(
                model_name="test",
                quantization="invalid",  # type: ignore
            )
            config.validate()

    def test_invalid_gpu_memory_rejected_before_load(self):
        """Test that invalid GPU memory utilization is caught at config level."""
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            config = ModelConfig(
                model_name="test",
                gpu_memory_utilization=1.5,  # Invalid: > 1.0
            )
            config.validate()

    def test_invalid_max_model_len_rejected_before_load(self):
        """Test that invalid max_model_len is caught at config level."""
        with pytest.raises(ValueError, match="max_model_len"):
            config = ModelConfig(
                model_name="test",
                max_model_len=100,  # Too small
            )
            config.validate()

    def test_valid_config_passes_to_loader_without_errors(self):
        """Test that valid config doesn't raise any errors during initialization."""
        # Create valid config
        config = ModelConfig(
            model_name="phi-3-mini",
            quantization="int4",
            device="auto",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )

        # Should validate without errors
        config.validate()

        # Should create loader without errors
        loader = ModelLoader(config=config)
        assert loader is not None


@pytest.mark.integration
class TestModelLoadingWithRealConfig:
    """Integration tests using real config objects (no mocks on config)."""

    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    def test_full_config_to_load_workflow(
        self,
        mock_mlx_generate,
        mock_mlx_load,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
    ):
        """Test complete workflow from config creation to model loading."""
        # 1. Create real config (no mocks)
        config = ModelConfig(
            model_name="phi-3-mini",
            quantization="4bit",
            device="auto",
            max_model_len=2048,
            warmup=True,
        )

        # 2. Validate config
        config.validate()

        # 3. Setup mocks for loading
        mock_device_detector.validate_platform.return_value = (True, "MPS available")
        mock_device_detector.detect.return_value = "mps"

        mock_get_model_spec.return_value = {
            "path": "mlx-community/phi-3-mini-4k-instruct",
            "context_length": 4096,
        }

        mock_memory_manager.can_fit_model.return_value = (True, "Fits in memory")
        mock_memory_manager.get_model_memory_usage.return_value = 3500

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        # Mock warmup generation
        mock_mlx_generate.return_value = "warmup output"

        # 4. Create loader
        loader = ModelLoader(config=config)

        # 5. Load model
        model_info = loader.load()

        # 6. Verify result
        assert model_info is not None
        assert model_info.model_name == "phi-3-mini"
        assert model_info.quantization == "4bit"
        assert model_info.device == "mps"
        assert model_info.is_loaded is True

        # 7. Verify warmup was called (because config.warmup=True)
        mock_mlx_generate.assert_called_once()

    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    @patch('mlx_lm.load')
    def test_config_lazy_load_defers_loading(
        self,
        mock_mlx_load,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
    ):
        """Test that lazy_load config option is respected."""
        # Create config with lazy_load
        config = ModelConfig(
            model_name="test-model",
            lazy_load=True,
        )

        # Create loader (should not load yet)
        loader = ModelLoader(config=config)

        # Verify not loaded
        assert not loader.is_loaded()

        # Now explicitly load
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())

        loader.load()
        assert loader.is_loaded()
