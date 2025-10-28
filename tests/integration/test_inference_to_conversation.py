"""
Integration tests for InferenceEngine → ConversationManager workflow.

Tests the integration between inference engine and conversation management,
ensuring generated text is properly integrated into conversation history.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging

from src.conversation.history import ConversationManager, Message
from src.inference.engine import InferenceEngine, GenerationResult
from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


class TestInferenceToConversationIntegration:
    """Test integration between InferenceEngine and ConversationManager."""

    def test_conversation_formats_prompt_for_inference(self):
        """Test that conversation creates proper prompt for inference."""
        conv = ConversationManager(max_context_tokens=2048)

        # Add messages
        conv.add_message("system", "You are a helpful assistant.")
        conv.add_message("user", "What is Python?")

        # Format as prompt
        prompt = conv.format_prompt(template="chatml")

        # Verify prompt contains messages
        assert "system" in prompt or "You are a helpful assistant" in prompt
        assert "user" in prompt or "What is Python?" in prompt

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_generated_text_added_to_conversation(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
        mock_model_config,
    ):
        """Test that generated text can be added to conversation."""
        # Setup inference
        from src.models.loader import ModelLoader

        mock_device_detector.validate_platform.return_value = (True, "Valid")
        mock_device_detector.detect.return_value = "mps"
        mock_get_model_spec.return_value = {
            "path": "test/model",
            "context_length": 2048,
        }
        mock_memory_manager.can_fit_model.return_value = (True, "Fits")
        mock_memory_manager.get_model_memory_usage.return_value = 2000
        mock_mlx_load.return_value = (MagicMock(), MagicMock())
        mock_mlx_generate.return_value = "Python is a programming language."

        loader = ModelLoader(config=mock_model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Create conversation
        conv = ConversationManager()
        conv.add_message("user", "What is Python?")

        # Generate response
        result = engine.generate(prompt=conv.format_prompt())

        # Add response to conversation
        conv.add_message("assistant", result.text)

        # Verify conversation state
        messages = conv.get_history()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is Python?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Python is a programming language."

    def test_multi_turn_conversation_workflow(self):
        """Test multi-turn conversation with mocked inference."""
        # Create conversation
        conv = ConversationManager()

        # Turn 1
        conv.add_message("user", "Hello")
        # Simulate inference
        conv.add_message("assistant", "Hi! How can I help?")

        # Turn 2
        conv.add_message("user", "Tell me about AI")
        # Simulate inference
        conv.add_message("assistant", "AI stands for Artificial Intelligence...")

        # Verify conversation
        messages = conv.get_history()
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[3].role == "assistant"

    def test_conversation_token_tracking_with_inference(self):
        """Test that token counts are tracked correctly."""
        conv = ConversationManager(max_context_tokens=1000)

        # Add messages
        msg1 = conv.add_message("user", "Short message")
        msg2 = conv.add_message("assistant", "Short response")

        # Verify tokens are counted
        assert msg1.tokens > 0
        assert msg2.tokens > 0
        assert conv.get_token_count() == msg1.tokens + msg2.tokens

    def test_conversation_context_window_truncation(self):
        """Test that conversation truncates when exceeding context window."""
        # Small context window
        conv = ConversationManager(max_context_tokens=100)

        # Add system message (should be preserved)
        conv.add_message("system", "You are helpful.")

        # Add many messages to exceed window
        for i in range(20):
            conv.add_message("user", f"Message {i} with lots of text" * 5)
            conv.add_message("assistant", f"Response {i} with lots of text" * 5)

        # Truncate to fit
        removed = conv.truncate_to_fit(reserve_tokens=10)

        # Verify truncation occurred
        assert removed > 0
        assert conv.get_token_count() < 90  # 100 - 10 reserve

        # Verify system message preserved
        messages = conv.get_history()
        assert any(msg.role == "system" for msg in messages)


class TestPromptFormattingForInference:
    """Test that different prompt formats work with inference."""

    def test_chatml_format_for_inference(self):
        """Test ChatML format prompt generation."""
        conv = ConversationManager()
        conv.add_message("system", "You are helpful.")
        conv.add_message("user", "Hello")

        prompt = conv.format_prompt(template="chatml")

        # ChatML uses special tokens
        assert len(prompt) > 0
        # Implementation-specific checks would go here

    def test_alpaca_format_for_inference(self):
        """Test Alpaca format prompt generation."""
        conv = ConversationManager()
        conv.add_message("user", "What is 2+2?")

        prompt = conv.format_prompt(template="alpaca")

        assert len(prompt) > 0
        # Alpaca-specific format checks

    def test_vicuna_format_for_inference(self):
        """Test Vicuna format prompt generation."""
        conv = ConversationManager()
        conv.add_message("user", "Tell me a joke")

        prompt = conv.format_prompt(template="vicuna")

        assert len(prompt) > 0
        # Vicuna-specific format checks


class TestConversationPersistenceWithInference:
    """Test that conversations can be saved/loaded during inference."""

    def test_save_and_load_conversation(self, tmp_path):
        """Test saving and loading conversation state."""
        # Create conversation
        conv = ConversationManager()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        conv.add_message("user", "How are you?")

        # Save to file
        filepath = tmp_path / "conversation.json"
        conv.save(str(filepath))

        # Load from file
        loaded_conv = ConversationManager.load(str(filepath))

        # Verify loaded state
        assert loaded_conv.get_message_count() == conv.get_message_count()
        assert loaded_conv.get_token_count() == conv.get_token_count()

        # Verify messages match
        original_msgs = conv.get_history()
        loaded_msgs = loaded_conv.get_history()

        for orig, loaded in zip(original_msgs, loaded_msgs):
            assert orig.role == loaded.role
            assert orig.content == loaded.content

    def test_resume_conversation_after_load(self, tmp_path):
        """Test that conversation can be resumed after loading."""
        # Create and save initial conversation
        conv1 = ConversationManager()
        conv1.add_message("user", "First message")
        conv1.add_message("assistant", "First response")

        filepath = tmp_path / "conv.json"
        conv1.save(str(filepath))

        # Load conversation
        conv2 = ConversationManager.load(str(filepath))

        # Continue conversation
        conv2.add_message("user", "Second message")
        conv2.add_message("assistant", "Second response")

        # Verify continuation
        assert conv2.get_message_count() == 4
        messages = conv2.get_history()
        assert messages[-1].content == "Second response"


@pytest.mark.integration
class TestFullInferenceConversationWorkflow:
    """End-to-end tests for inference with conversation management."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_complete_conversation_workflow(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test complete multi-turn conversation with inference."""
        from src.config.model_config import ModelConfig
        from src.models.loader import ModelLoader

        # Setup
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

        # Different responses for each turn
        responses = [
            "Hello! How can I help?",
            "Python is a programming language.",
            "It was created by Guido van Rossum.",
        ]
        mock_mlx_generate.side_effect = responses

        # Load model and create engine
        loader = ModelLoader(config=config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Create conversation
        conv = ConversationManager(max_context_tokens=4096)
        conv.add_message("system", "You are a helpful assistant.")

        # Turn 1
        conv.add_message("user", "Hello")
        prompt1 = conv.format_prompt()
        result1 = engine.generate(prompt=prompt1)
        conv.add_message("assistant", result1.text)

        # Turn 2
        conv.add_message("user", "What is Python?")
        prompt2 = conv.format_prompt()
        result2 = engine.generate(prompt=prompt2)
        conv.add_message("assistant", result2.text)

        # Turn 3
        conv.add_message("user", "Who created it?")
        prompt3 = conv.format_prompt()
        result3 = engine.generate(prompt=prompt3)
        conv.add_message("assistant", result3.text)

        # Verify conversation state
        messages = conv.get_history()
        assert len(messages) == 7  # system + 3 user + 3 assistant
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Hello! How can I help?"
        assert messages[4].content == "Python is a programming language."
        assert messages[6].content == "It was created by Guido van Rossum."

    @patch('mlx_lm.load')
    @patch('mlx_lm.stream_generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_streaming_conversation_workflow(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_stream_generate,
        mock_mlx_load,
    ):
        """Test streaming generation integrated with conversation."""
        from src.config.model_config import ModelConfig
        from src.models.loader import ModelLoader

        # Setup
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

        # Create conversation
        conv = ConversationManager()
        conv.add_message("user", "Hi")

        # Stream generation
        prompt = conv.format_prompt()
        tokens = []
        for token_dict in engine.generate_stream(prompt=prompt):
            # Extract token string from dict (new format)
            token = token_dict['token'] if isinstance(token_dict, dict) else token_dict
            tokens.append(token)

        # Add streamed response to conversation
        full_response = "".join(tokens)
        conv.add_message("assistant", full_response)

        # Verify
        messages = conv.get_history()
        assert len(messages) == 2
        assert messages[1].content == "Hello there!"

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_conversation_with_context_truncation(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test that conversation properly truncates when approaching limits."""
        from src.config.model_config import ModelConfig
        from src.models.loader import ModelLoader

        # Setup with small context window
        config = ModelConfig(model_name="test", max_model_len=500, warmup=False)
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

        # Load model and create engine
        loader = ModelLoader(config=config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Create conversation with small window
        conv = ConversationManager(max_context_tokens=500)
        conv.add_message("system", "You are helpful.")

        # Add many messages
        for i in range(15):
            conv.add_message("user", f"Long message {i} " * 20)
            # Truncate before generating to fit
            conv.truncate_to_fit(reserve_tokens=100)

            prompt = conv.format_prompt()
            result = engine.generate(prompt=prompt)
            conv.add_message("assistant", result.text)

            # Truncate again after response
            conv.truncate_to_fit(reserve_tokens=100)

        # Verify conversation stayed within limits
        assert conv.get_token_count() < 500
        # System message should still be there
        messages = conv.get_history()
        assert any(msg.role == "system" for msg in messages)
