"""
End-to-end tests with fully mocked model.

Comprehensive E2E testing without requiring real MLX models.
Tests all CLI commands, error handling, and multi-turn conversations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import logging
from pathlib import Path

from src.config.model_config import ModelConfig
from src.config.inference_config import InferenceConfig
from src.config.cli_config import CLIConfig
from src.models.loader import ModelLoader
from src.inference.engine import InferenceEngine
from src.conversation.history import ConversationManager
from src.cli.repl import REPL
from src.cli.commands import CommandDispatcher
from src.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_complete_stack():
    """Fixture providing a complete mocked stack for E2E testing."""
    with patch('mlx_lm.load') as mock_load, \
         patch('mlx_lm.generate') as mock_generate, \
         patch('mlx_lm.stream_generate') as mock_stream, \
         patch('src.devices.DeviceDetector') as mock_detector, \
         patch('src.models.registry.get_model_spec') as mock_spec, \
         patch('src.models.memory.MemoryManager') as mock_memory:

        # Setup all mocks
        mock_detector.validate_platform.return_value = (True, "Platform valid")
        mock_detector.detect.return_value = "mps"
        mock_spec.return_value = {
            "path": "test/model",
            "context_length": 4096,
        }
        mock_memory.can_fit_model.return_value = (True, "Fits")
        mock_memory.get_model_memory_usage.return_value = 3000
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.return_value = "Mock response"

        yield {
            'load': mock_load,
            'generate': mock_generate,
            'stream': mock_stream,
            'detector': mock_detector,
            'spec': mock_spec,
            'memory': mock_memory,
        }


@pytest.mark.integration
class TestE2EMockFullWorkflow:
    """Test complete workflow with fully mocked model."""

    def test_full_chat_session(self, mock_complete_stack):
        """Test a complete multi-turn chat session."""
        # Create configs
        model_config = ModelConfig(model_name="phi-3-mini", warmup=False)
        inference_config = InferenceConfig(temperature=0.7, max_tokens=100)
        cli_config = CLIConfig(colorize=False, verbose=False)

        # Setup mock responses
        mock_complete_stack['generate'].side_effect = [
            "Hi! How can I help you today?",
            "Python is a high-level programming language.",
            "It's great for data science, web development, and automation.",
            "You're welcome! Feel free to ask more questions.",
        ]

        # Initialize stack
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader, config=inference_config)
        conversation = ConversationManager(max_context_tokens=4096)
        conversation.add_message("system", "You are a helpful assistant.")
        repl = REPL(conversation=conversation, engine=engine, config=cli_config)

        # Simulate chat session
        user_messages = [
            "Hello",
            "What is Python?",
            "What can I use it for?",
            "Thanks!",
        ]

        for user_msg in user_messages:
            repl._process_message(user_msg)

        # Verify conversation
        messages = conversation.get_history()
        assert len(messages) == 9  # system + 4 user + 4 assistant
        assert messages[0].role == "system"

        # Verify all user messages present
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) == 4
        assert [m.content for m in user_msgs] == user_messages

        # Verify all assistant responses present
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) == 4

    def test_streaming_chat_session(self, mock_complete_stack):
        """Test chat session with streaming responses."""
        # Setup streaming responses
        stream1 = iter(["Hello", "!", " How", " can", " I", " help", "?"])
        stream2 = iter(["Python", " is", " awesome", "!"])
        mock_complete_stack['stream'].side_effect = [stream1, stream2]

        # Initialize
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        # Stream first response
        conversation.add_message("user", "Hi")
        tokens1 = list(engine.generate_stream(prompt=conversation.format_prompt()))
        # Extract text from dict tokens (new format)
        response1 = "".join(t['token'] if isinstance(t, dict) else t for t in tokens1)
        conversation.add_message("assistant", response1)

        # Stream second response
        conversation.add_message("user", "Tell me about Python")
        tokens2 = list(engine.generate_stream(prompt=conversation.format_prompt()))
        # Extract text from dict tokens (new format)
        response2 = "".join(t['token'] if isinstance(t, dict) else t for t in tokens2)
        conversation.add_message("assistant", response2)

        # Verify
        assert conversation.get_message_count() == 4
        assert response1 == "Hello! How can I help?"
        assert response2 == "Python is awesome!"


@pytest.mark.integration
class TestE2EMockCLICommands:
    """Test all CLI commands in E2E context."""

    def test_help_command(self, mock_complete_stack):
        """Test /help command."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=engine)

        # Execute /help
        continue_loop, output = repl.dispatcher.execute("/help")

        assert continue_loop is True
        assert output is not None
        assert isinstance(output, str)

    def test_clear_command(self, mock_complete_stack):
        """Test /clear command."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Add messages
        conversation.add_message("user", "Hello")
        conversation.add_message("assistant", "Hi")
        assert conversation.get_message_count() == 2

        # Execute /clear
        continue_loop, output = repl.dispatcher.execute("/clear")

        assert continue_loop is True
        assert conversation.get_message_count() == 0

    def test_history_command(self, mock_complete_stack):
        """Test /history command."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        conversation.add_message("user", "Test message")
        conversation.add_message("assistant", "Test response")

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Execute /history
        continue_loop, output = repl.dispatcher.execute("/history")

        assert continue_loop is True
        assert output is not None

    def test_save_command(self, mock_complete_stack, tmp_path):
        """Test /save command."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        conversation.add_message("user", "Save this")

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Execute /save
        filepath = tmp_path / "test_save.json"
        continue_loop, output = repl.dispatcher.execute(f"/save {filepath}")

        assert continue_loop is True
        assert filepath.exists()

        # Verify saved content
        loaded = ConversationManager.load(str(filepath))
        assert loaded.get_message_count() == 1

    def test_load_command(self, mock_complete_stack, tmp_path):
        """Test /load command."""
        # Create and save a conversation
        conv1 = ConversationManager()
        conv1.add_message("user", "Loaded message")
        filepath = tmp_path / "test_load.json"
        conv1.save(str(filepath))

        # Create new session
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Execute /load
        continue_loop, output = repl.dispatcher.execute(f"/load {filepath}")

        assert continue_loop is True
        assert conversation.get_message_count() == 1
        assert conversation.get_history()[0].content == "Loaded message"

    def test_info_command(self, mock_complete_stack):
        """Test /info command."""
        model_config = ModelConfig(model_name="phi-3-mini", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.model_loader = loader

        # Execute /info
        continue_loop, output = repl.dispatcher.execute("/info")

        assert continue_loop is True
        assert output is not None

    def test_exit_command(self, mock_complete_stack):
        """Test /exit command."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        repl = REPL(conversation=conversation, engine=engine)

        # Execute /exit
        continue_loop, output = repl.dispatcher.execute("/exit")

        assert continue_loop is False


@pytest.mark.integration
class TestE2EMockErrorHandling:
    """Test error handling in E2E context."""

    def test_generation_error_handling(self, mock_complete_stack):
        """Test that generation errors are handled gracefully."""
        from src.inference.exceptions import GenerationError

        # Setup to fail
        mock_complete_stack['generate'].side_effect = Exception("Generation failed")

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=engine)

        # Should handle error gracefully
        with patch('src.cli.output.print_error') as mock_error:
            result = repl._process_message("Test")
            assert result is True  # Should continue
            mock_error.assert_called()

    def test_save_to_invalid_path(self, mock_complete_stack):
        """Test saving conversation to invalid path."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Try to save to invalid path
        with patch('src.cli.output.print_error'):
            continue_loop, output = repl.dispatcher.execute("/save /invalid/path/file.json")
            # Should not crash
            assert continue_loop is True

    def test_load_nonexistent_file(self, mock_complete_stack):
        """Test loading conversation from non-existent file."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation

        # Try to load non-existent file
        with patch('src.cli.output.print_error'):
            continue_loop, output = repl.dispatcher.execute("/load /nonexistent/file.json")
            # Should not crash
            assert continue_loop is True


@pytest.mark.integration
class TestE2EMockComplexScenarios:
    """Test complex scenarios in E2E context."""

    def test_very_long_conversation(self, mock_complete_stack):
        """Test handling of very long conversations."""
        mock_complete_stack['generate'].return_value = "Response"

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager(max_context_tokens=1000)
        repl = REPL(conversation=conversation, engine=engine)

        # Add many messages
        for i in range(50):
            repl._process_message(f"Message {i}")

            # Periodically truncate
            if i % 5 == 0:
                conversation.truncate_to_fit(reserve_tokens=100)

        # Should still be within limits
        assert conversation.get_token_count() < 1000

    def test_rapid_save_load_cycles(self, mock_complete_stack, tmp_path):
        """Test rapid save/load cycles."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        for i in range(5):
            # Create conversation
            conv = ConversationManager()
            conv.add_message("user", f"Message {i}")

            # Save
            filepath = tmp_path / f"conv_{i}.json"
            conv.save(str(filepath))

            # Load
            loaded_conv = ConversationManager.load(str(filepath))
            assert loaded_conv.get_message_count() == 1

    def test_conversation_with_all_commands(self, mock_complete_stack, tmp_path):
        """Test using all commands in sequence during a conversation."""
        mock_complete_stack['generate'].return_value = "Response"

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=engine)
        repl.dispatcher.conversation = conversation
        repl.dispatcher.model_loader = loader

        # Chat
        repl._process_message("Hello")
        assert conversation.get_message_count() == 2

        # /history
        repl.dispatcher.execute("/history")

        # /info
        repl.dispatcher.execute("/info")

        # /save
        filepath = tmp_path / "all_commands.json"
        repl.dispatcher.execute(f"/save {filepath}")

        # /clear
        repl.dispatcher.execute("/clear")
        assert conversation.get_message_count() == 0

        # /load
        repl.dispatcher.execute(f"/load {filepath}")
        assert conversation.get_message_count() == 2

        # Chat again
        repl._process_message("Another message")
        assert conversation.get_message_count() == 4

        # /help
        repl.dispatcher.execute("/help")

        # Everything should work without errors

    def test_context_window_with_system_message(self, mock_complete_stack):
        """Test that system message is preserved during context truncation."""
        mock_complete_stack['generate'].return_value = "Response"

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager(max_context_tokens=500)
        conversation.add_message("system", "Important system prompt")

        repl = REPL(conversation=conversation, engine=engine)

        # Add many messages to force truncation
        for i in range(30):
            repl._process_message(f"Long message {i} " * 10)
            conversation.truncate_to_fit(reserve_tokens=100)

        # System message should still be there
        messages = conversation.get_history()
        system_msgs = [m for m in messages if m.role == "system"]
        assert len(system_msgs) >= 1
        assert system_msgs[0].content == "Important system prompt"

    def test_different_prompt_templates(self, mock_complete_stack):
        """Test conversation with different prompt templates."""
        mock_complete_stack['generate'].return_value = "Response"

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        templates = ["chatml", "alpaca", "vicuna"]

        for template in templates:
            conversation = ConversationManager()
            conversation.add_message("user", "Test")

            # Format with template
            prompt = conversation.format_prompt(template=template)
            assert len(prompt) > 0

            # Generate (should work with any template)
            result = engine.generate(prompt=prompt)
            assert result.text == "Response"

    def test_metrics_tracking_across_session(self, mock_complete_stack):
        """Test that metrics are properly tracked across a session."""
        mock_complete_stack['generate'].return_value = "Response"

        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=engine)

        # Multiple generations
        for i in range(5):
            repl._process_message(f"Message {i}")

        # Check metrics
        metrics = engine.get_metrics_summary()
        assert metrics['total_generations'] == 5

    @patch('src.cli.input.create_input_handler')
    def test_autocomplete_functionality(self, mock_input_handler, mock_complete_stack):
        """Test command autocomplete functionality."""
        model_config = ModelConfig(model_name="test", warmup=False)
        loader = ModelLoader(config=model_config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)
        conversation = ConversationManager()

        # Create REPL (triggers input handler creation)
        repl = REPL(conversation=conversation, engine=engine)

        # Verify input handler was created with commands
        mock_input_handler.assert_called_once()
        call_kwargs = mock_input_handler.call_args.kwargs
        assert 'commands' in call_kwargs

        # Verify commands list is populated
        commands = call_kwargs['commands']
        assert len(commands) > 0
        # Should include standard commands (with or without /)
        assert any('help' in cmd.lower() for cmd in commands)
        assert any('clear' in cmd.lower() for cmd in commands)
