"""
End-to-end workflow integration tests.

Tests the complete pipeline from configuration to CLI output,
ensuring all components work together correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from src.config.model_config import ModelConfig
from src.config.inference_config import InferenceConfig
from src.config.cli_config import CLIConfig
from src.models.loader import ModelLoader
from src.inference.engine import InferenceEngine
from src.conversation.history import ConversationManager
from src.cli.repl import REPL
from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestFullE2EWorkflow:
    """Test complete end-to-end workflow with all components."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_complete_pipeline_config_to_response(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """
        Test complete pipeline: Config → Model → Inference → Conversation → CLI.

        This is the golden path test that validates all components work together.
        """
        # ===== PHASE 1: Configuration =====
        model_config = ModelConfig(
            model_name="phi-3-mini",
            quantization="int4",
            device="auto",
            max_model_len=2048,
            warmup=False,
        )

        inference_config = InferenceConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            streaming=False,
        )

        cli_config = CLIConfig(
            colorize=True,
            show_tokens=True,
            verbose=False,
        )

        # Validate all configs
        model_config.validate()
        inference_config.validate()

        # ===== PHASE 2: Model Loading =====
        mock_device_detector.validate_platform.return_value = (True, "MPS available")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "mlx-community/phi-3-mini-4k-instruct",
            "context_length": 4096,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Model fits in memory")
        mock_memory_manager.get_model_memory_usage.return_value = 3500

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)

        loader = ModelLoader(config=model_config)
        model_info = loader.load()

        assert model_info.is_loaded
        assert model_info.model_name == "phi-3-mini"

        # ===== PHASE 3: Inference Engine =====
        engine = InferenceEngine(model_loader=loader, config=inference_config)
        assert engine is not None

        # ===== PHASE 4: Conversation Management =====
        conversation = ConversationManager(
            max_context_tokens=model_config.max_model_len
        )
        conversation.add_message("system", "You are a helpful AI assistant.")

        # ===== PHASE 5: REPL/CLI =====
        repl = REPL(
            conversation=conversation,
            engine=engine,
            config=cli_config,
        )
        assert repl.conversation == conversation
        assert repl.engine == engine

        # ===== PHASE 6: User Interaction =====
        mock_mlx_generate.side_effect = [
            "Hello! I'm here to help you with any questions.",
            "Python is a high-level, interpreted programming language.",
            "It was created by Guido van Rossum and first released in 1991.",
        ]

        # Turn 1: Greeting
        repl._process_message("Hello!")
        messages = conversation.get_history()
        assert len(messages) == 3  # system + user + assistant
        assert messages[1].role == "user"
        assert messages[1].content == "Hello!"
        assert messages[2].role == "assistant"

        # Turn 2: Question about Python
        repl._process_message("What is Python?")
        messages = conversation.get_history()
        assert len(messages) == 5
        assert messages[3].content == "What is Python?"
        assert "Python" in messages[4].content

        # Turn 3: Follow-up question
        repl._process_message("Who created it?")
        messages = conversation.get_history()
        assert len(messages) == 7
        assert "Guido" in messages[6].content or "1991" in messages[6].content

        # ===== VERIFICATION =====
        # Verify conversation state
        assert conversation.get_message_count() == 7
        assert conversation.get_token_count() > 0

        # Verify model still loaded
        assert loader.is_loaded()

        # Verify metrics tracked
        metrics = engine.get_metrics_summary()
        assert metrics['total_generations'] == 3

    @patch('mlx_lm.load')
    @patch('mlx_lm.stream_generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_streaming_e2e_workflow(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_stream_generate,
        mock_mlx_load,
    ):
        """Test end-to-end workflow with streaming generation."""
        # Setup
        model_config = ModelConfig(model_name="test", warmup=False)
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())

        # Mock streaming responses
        stream1 = iter(["Hello", "!", " How", " can", " I", " help", "?"])
        stream2 = iter(["Python", " is", " great", "."])
        mock_stream_generate.side_effect = [stream1, stream2]

        # Load model and create components
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        # Stream first response
        prompt1 = "Hi"
        conversation.add_message("user", prompt1)
        tokens1 = list(engine.generate_stream(prompt=conversation.format_prompt()))
        response1 = "".join(tokens1)
        conversation.add_message("assistant", response1)

        # Stream second response
        prompt2 = "Tell me about Python"
        conversation.add_message("user", prompt2)
        tokens2 = list(engine.generate_stream(prompt=conversation.format_prompt()))
        response2 = "".join(tokens2)
        conversation.add_message("assistant", response2)

        # Verify
        assert conversation.get_message_count() == 4
        assert response1 == "Hello! How can I help?"
        assert response2 == "Python is great."

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_with_conversation_persistence(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        tmp_path,
    ):
        """Test E2E workflow with save/load conversation."""
        # Setup
        model_config = ModelConfig(model_name="test", warmup=False)
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.side_effect = ["Response 1", "Response 2"]

        # Session 1: Initial conversation
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conv1 = ConversationManager()
        repl1 = REPL(conversation=conv1, engine=engine)

        repl1._process_message("First message")
        assert conv1.get_message_count() == 2

        # Save conversation
        save_path = tmp_path / "conversation.json"
        conv1.save(str(save_path))

        # Session 2: Load and continue
        conv2 = ConversationManager.load(str(save_path))
        repl2 = REPL(conversation=conv2, engine=engine)

        repl2._process_message("Second message")
        assert conv2.get_message_count() == 4

        # Verify continuity
        messages = conv2.get_history()
        assert messages[0].content == "First message"
        assert messages[1].content == "Response 1"
        assert messages[2].content == "Second message"
        assert messages[3].content == "Response 2"

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_with_context_management(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test E2E workflow with automatic context window management."""
        # Setup with small context window
        model_config = ModelConfig(
            model_name="test",
            max_model_len=500,  # Small window
            warmup=False,
        )
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 500,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.return_value = "Response"

        # Create components
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager(max_context_tokens=500)
        conversation.add_message("system", "You are helpful.")

        # Add many messages
        for i in range(10):
            message = f"This is a long user message {i} " * 10
            conversation.add_message("user", message)

            # Truncate before generating
            conversation.truncate_to_fit(reserve_tokens=100)

            # Generate response
            prompt = conversation.format_prompt()
            result = engine.generate(prompt=prompt)
            conversation.add_message("assistant", result.text)

            # Truncate after response
            conversation.truncate_to_fit(reserve_tokens=100)

        # Verify context stayed within limits
        assert conversation.get_token_count() < 500

        # Verify system message preserved
        messages = conversation.get_history()
        assert any(msg.role == "system" for msg in messages)


@pytest.mark.integration
class TestE2EErrorHandling:
    """Test end-to-end error handling scenarios."""

    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_handles_model_load_failure(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
    ):
        """Test that E2E workflow gracefully handles model loading failures."""
        from src.models.exceptions import ModelLoadError

        # Setup to fail
        model_config = ModelConfig(model_name="test")
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = None  # Model not found

        loader = ModelLoader(config=model_config)

        # Should raise appropriate error
        with pytest.raises(Exception):  # ModelNotFoundError
            loader.load()

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_handles_generation_failure(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test that E2E workflow handles generation failures gracefully."""
        from src.inference.exceptions import GenerationError

        # Setup
        model_config = ModelConfig(model_name="test", warmup=False)
        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())

        # Make generation fail
        mock_mlx_generate.side_effect = Exception("Generation failed")

        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Generation should raise error
        with pytest.raises(GenerationError):
            engine.generate(prompt="Test")


@pytest.mark.integration
class TestE2EConfigurationVariations:
    """Test E2E workflow with different configurations."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_with_different_quantizations(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test E2E workflow with different quantization levels."""
        quantizations = ["int4", "int8", "fp16"]

        for quant in quantizations:
            # Setup
            model_config = ModelConfig(
                model_name="test",
                quantization=quant,
                warmup=False,
            )
            mock_device_detector.validate_platform.return_value = (True, "Valid")
            mock_device_detector.detect.return_value = "mps"
            mock_get_model_spec.return_value = {
                "path": "test/model",
                "context_length": 2048,
            }
            mock_memory_manager.can_fit_model.return_value = (True, "Fits")
            mock_memory_manager.get_model_memory_usage.return_value = 2000
            mock_mlx_load.return_value = (MagicMock(), MagicMock())
            mock_mlx_generate.return_value = f"Response with {quant}"

            # Run workflow
            loader = ModelLoader(config=model_config)
            model_info = loader.load()
            assert model_info.quantization == quant

            engine = InferenceEngine(model_loader=loader)
            result = engine.generate(prompt="Test")
            assert quant in result.text

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_e2e_with_different_sampling_params(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test E2E workflow with various sampling parameters."""
        # Setup
        model_config = ModelConfig(model_name="test", warmup=False)
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

        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Test different sampling parameters
        sampling_configs = [
            {"temperature": 0.5, "max_tokens": 50},
            {"temperature": 0.9, "max_tokens": 100},
            {"temperature": 1.0, "top_p": 0.95, "max_tokens": 200},
        ]

        for config in sampling_configs:
            params = SamplingParams(**config)
            result = engine.generate(prompt="Test", sampling_params=params)
            assert result.text == "Response"
