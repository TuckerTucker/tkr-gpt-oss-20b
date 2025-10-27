"""
Integration tests for ModelLoader → InferenceEngine workflow.

Tests the integration between model loading and inference engine,
ensuring models are properly passed and used for text generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from src.models.loader import ModelLoader, ModelInfo
from src.inference.engine import InferenceEngine, GenerationResult
from src.inference.exceptions import ModelNotLoadedError, GenerationError
from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


class TestModelLoaderToInferenceIntegration:
    """Test integration between ModelLoader and InferenceEngine."""

    def test_inference_engine_requires_loaded_model(self, mock_model_config):
        """Test that InferenceEngine requires a loaded model."""
        # Create unloaded model loader
        loader = ModelLoader(config=mock_model_config)

        # Should raise because model isn't loaded
        with pytest.raises(ModelNotLoadedError):
            InferenceEngine(model_loader=loader)

    @patch('mlx_lm.load')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_inference_engine_accepts_loaded_model(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that InferenceEngine accepts a properly loaded model."""
        # Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        # Load model
        loader = ModelLoader(config=mock_model_config)
        loader.load()

        # Should succeed with loaded model
        engine = InferenceEngine(model_loader=loader)
        assert engine is not None
        assert engine.model_loader == loader

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_inference_uses_loaded_model(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that InferenceEngine uses the model from ModelLoader."""
        # Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        # Mock generation output
        mock_mlx_generate.return_value = "Generated response text"

        # Load model
        loader = ModelLoader(config=mock_model_config)
        loader.load()

        # Create engine
        engine = InferenceEngine(model_loader=loader)

        # Generate text
        result = engine.generate(prompt="Test prompt")

        # Verify model was used
        assert mock_mlx_generate.called
        assert result.text == "Generated response text"

    @patch('mlx_lm.load')
    @patch('mlx_lm.stream_generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_inference_streaming_uses_loaded_model(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_stream_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that streaming generation uses the loaded model."""
        # Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        # Mock streaming output
        mock_stream_generate.return_value = iter(["Hello", " world", "!"])

        # Load model
        loader = ModelLoader(config=mock_model_config)
        loader.load()

        # Create engine
        engine = InferenceEngine(model_loader=loader)

        # Stream text
        tokens = list(engine.generate_stream(prompt="Test prompt"))

        # Verify streaming worked
        assert mock_stream_generate.called
        assert tokens == ["Hello", " world", "!"]

    @patch('mlx_lm.load')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_inference_fails_if_model_unloaded(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that inference fails if model is unloaded after engine creation."""
        # Setup and load
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())

        loader = ModelLoader(config=mock_model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Unload model
        loader.unload()

        # Generation should fail
        with pytest.raises(ModelNotLoadedError):
            engine.generate(prompt="Test")


class TestSamplingParamsIntegration:
    """Test integration of SamplingParams with inference."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_sampling_params_passed_to_generation(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that SamplingParams are properly passed to generation."""
        # Setup
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.return_value = "Response"

        # Load and create engine
        loader = ModelLoader(config=mock_model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Create custom sampling params
        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=100,
            repetition_penalty=1.2,
        )

        # Generate
        engine.generate(prompt="Test", sampling_params=sampling_params)

        # Verify mlx_lm.generate was called with correct params
        call_kwargs = mock_mlx_generate.call_args.kwargs
        assert call_kwargs['temp'] == 0.9
        assert call_kwargs['max_tokens'] == 100
        assert call_kwargs['top_p'] == 0.95
        assert call_kwargs['repetition_penalty'] == 1.2

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_default_sampling_params_used_when_none_provided(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that default sampling params are used when none provided."""
        # Setup
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.return_value = "Response"

        # Load and create engine
        loader = ModelLoader(config=mock_model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Generate without params
        engine.generate(prompt="Test")

        # Verify defaults were used
        call_kwargs = mock_mlx_generate.call_args.kwargs
        assert 'temp' in call_kwargs
        assert 'max_tokens' in call_kwargs


class TestModelSwitchingIntegration:
    """Test model switching with inference engine."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_engine_works_after_model_switch(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that inference engine works after model switching."""
        # Setup
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"

        def get_spec_side_effect(model_name):
            return {
                "path": f"test/{model_name}",
                "context_length": 2048,
            }

        mock_get_model_spec.side_effect = get_spec_side_effect
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000

        # Different models
        model1 = MagicMock()
        model2 = MagicMock()
        tokenizer = MagicMock()
        mock_mlx_load.side_effect = [(model1, tokenizer), (model2, tokenizer)]
        mock_mlx_generate.return_value = "Response"

        # Load initial model
        loader = ModelLoader(config=mock_model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Generate with first model
        result1 = engine.generate(prompt="Test 1")
        assert result1.text == "Response"

        # Switch model
        new_config = Mock()
        new_config.model_name = "new-model"
        new_config.quantization = "int4"
        new_config.device = "auto"
        new_config.max_model_len = 2048
        new_config.warmup = False
        new_config.model_path = None

        loader.switch_model(new_config)

        # Generate with new model (should still work)
        result2 = engine.generate(prompt="Test 2")
        assert result2.text == "Response"


@pytest.mark.integration
class TestFullModelToInferenceWorkflow:
    """End-to-end tests for model loading to inference."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_complete_workflow_config_to_generation(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test complete workflow from config to text generation."""
        # 1. Create config
        from src.config.model_config import ModelConfig

        config = ModelConfig(
            model_name="phi-3-mini",
            quantization="int4",
            device="auto",
            warmup=False,
        )

        # 2. Setup mocks
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "mlx-community/phi-3-mini",
            "context_length": 4096,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 3000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.return_value = "This is a generated response."

        # 3. Load model
        loader = ModelLoader(config=config)
        model_info = loader.load()

        assert model_info.is_loaded

        # 4. Create inference engine
        engine = InferenceEngine(model_loader=loader)

        # 5. Generate text
        result = engine.generate(
            prompt="Hello, how are you?",
            sampling_params=SamplingParams(
                temperature=0.7,
                max_tokens=50,
            )
        )

        # 6. Verify result
        assert result.text == "This is a generated response."
        assert result.tokens_generated > 0
        assert result.latency_ms > 0
        assert result.finish_reason in ["stop", "length"]

    @patch('mlx_lm.load')
    @patch('mlx_lm.stream_generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_complete_streaming_workflow(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_stream_generate,
        mock_mlx_load,
    ):
        """Test complete streaming workflow."""
        # Setup
        from src.config.model_config import ModelConfig

        config = ModelConfig(model_name="test", warmup=False)

        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_stream_generate.return_value = iter(["Hello", " there", "!"])

        # Load and create engine
        loader = ModelLoader(config=config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Stream generation
        tokens = []
        for token in engine.generate_stream(prompt="Test"):
            tokens.append(token)

        # Verify
        assert tokens == ["Hello", " there", "!"]
        assert len(tokens) == 3
