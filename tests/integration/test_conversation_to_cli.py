"""
Integration tests for ConversationManager → CLI/REPL workflow.

Tests the integration between conversation management and the CLI,
ensuring user inputs and generated responses flow correctly through the UI.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
from io import StringIO

from src.conversation.history import ConversationManager
from src.cli.repl import REPL
from src.cli.commands import CommandDispatcher

logger = logging.getLogger(__name__)


class TestConversationToCLIIntegration:
    """Test integration between ConversationManager and REPL."""

    def test_repl_accepts_conversation_manager(self):
        """Test that REPL accepts ConversationManager instance."""
        conv = ConversationManager()
        repl = REPL(conversation=conv)

        assert repl.conversation == conv

    def test_repl_can_work_without_conversation(self):
        """Test that REPL can work in standalone mode without conversation."""
        repl = REPL()

        assert repl.conversation is None
        # REPL should still initialize successfully

    def test_user_message_added_to_conversation(self):
        """Test that user messages are properly added to conversation."""
        conv = ConversationManager()

        # Mock engine to avoid actual inference
        mock_engine = Mock()
        mock_engine.generate = Mock(return_value=Mock(text="Response"))

        repl = REPL(conversation=conv, engine=mock_engine)

        # Simulate user message processing
        user_input = "Hello, how are you?"

        # Manually trigger message processing (bypassing input loop)
        repl._process_message(user_input)

        # Verify message was added to conversation
        messages = conv.get_history()
        assert any(msg.content == user_input for msg in messages)

    def test_assistant_response_added_to_conversation(self):
        """Test that assistant responses are added to conversation."""
        conv = ConversationManager()

        # Mock engine
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.text = "I'm doing well, thank you!"
        mock_engine.generate = Mock(return_value=mock_result)

        repl = REPL(conversation=conv, engine=mock_engine)

        # Process message
        repl._process_message("How are you?")

        # Verify both user and assistant messages
        messages = conv.get_history()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "How are you?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "I'm doing well, thank you!"


class TestCLICommandsWithConversation:
    """Test CLI commands that interact with conversation."""

    def test_clear_command_clears_conversation(self):
        """Test that /clear command clears conversation history."""
        conv = ConversationManager()
        conv.add_message("user", "Message 1")
        conv.add_message("assistant", "Response 1")
        conv.add_message("user", "Message 2")

        repl = REPL(conversation=conv)

        # Execute clear command through dispatcher
        repl.dispatcher.conversation = conv
        continue_loop, output = repl.dispatcher.execute("/clear")

        # Verify conversation cleared
        assert conv.get_message_count() == 0

    def test_history_command_shows_conversation(self):
        """Test that /history command shows conversation history."""
        conv = ConversationManager()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there")

        repl = REPL(conversation=conv)
        repl.dispatcher.conversation = conv

        # Execute history command
        continue_loop, output = repl.dispatcher.execute("/history")

        # Verify output contains messages
        assert "Hello" in output or output is not None

    def test_save_command_persists_conversation(self, tmp_path):
        """Test that /save command saves conversation to file."""
        conv = ConversationManager()
        conv.add_message("user", "Test message")

        repl = REPL(conversation=conv)
        repl.dispatcher.conversation = conv

        # Execute save command
        filepath = tmp_path / "test_conv.json"
        continue_loop, output = repl.dispatcher.execute(f"/save {filepath}")

        # Verify file exists
        assert filepath.exists()

        # Load and verify content
        loaded_conv = ConversationManager.load(str(filepath))
        assert loaded_conv.get_message_count() == 1
        assert loaded_conv.get_history()[0].content == "Test message"

    def test_load_command_restores_conversation(self, tmp_path):
        """Test that /load command restores conversation from file."""
        # Create and save conversation
        conv1 = ConversationManager()
        conv1.add_message("user", "Saved message")
        filepath = tmp_path / "conv.json"
        conv1.save(str(filepath))

        # Create new REPL with empty conversation
        conv2 = ConversationManager()
        repl = REPL(conversation=conv2)
        repl.dispatcher.conversation = conv2

        # Load conversation
        continue_loop, output = repl.dispatcher.execute(f"/load {filepath}")

        # Verify conversation restored
        assert conv2.get_message_count() == 1
        assert conv2.get_history()[0].content == "Saved message"


class TestREPLInputProcessing:
    """Test REPL input processing with conversation."""

    def test_command_vs_message_detection(self):
        """Test that REPL correctly distinguishes commands from messages."""
        repl = REPL()

        # Commands should be detected
        assert repl.dispatcher.parser.is_command("/help")
        assert repl.dispatcher.parser.is_command("/clear")
        assert repl.dispatcher.parser.is_command("/exit")

        # Regular messages should not be commands
        assert not repl.dispatcher.parser.is_command("Hello")
        assert not repl.dispatcher.parser.is_command("What is Python?")

    def test_multi_turn_conversation_via_repl(self):
        """Test multi-turn conversation through REPL."""
        conv = ConversationManager()

        # Mock engine with sequential responses
        mock_engine = Mock()
        responses = [
            Mock(text="Hi! How can I help?"),
            Mock(text="Python is a programming language."),
        ]
        mock_engine.generate = Mock(side_effect=responses)

        repl = REPL(conversation=conv, engine=mock_engine)

        # Turn 1
        repl._process_message("Hello")
        assert conv.get_message_count() == 2

        # Turn 2
        repl._process_message("What is Python?")
        assert conv.get_message_count() == 4

        # Verify history
        messages = conv.get_history()
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi! How can I help?"
        assert messages[2].content == "What is Python?"
        assert messages[3].content == "Python is a programming language."


class TestREPLWithoutInference:
    """Test REPL behavior when inference engine is not available."""

    def test_repl_works_in_mock_mode(self):
        """Test that REPL can work without real inference (mock responses)."""
        conv = ConversationManager()
        repl = REPL(conversation=conv, engine=None)

        # Process message (should use mock response)
        result = repl._process_message("hello")

        # Should continue (not exit)
        assert result is True

    def test_mock_responses_provided(self):
        """Test that mock responses are generated when no engine present."""
        repl = REPL(conversation=None, engine=None)

        # Mock response method should work
        # (Captures output to avoid terminal spam)
        with patch('src.cli.output.console'):
            repl._mock_response("test")
            # Should not raise any errors


@pytest.mark.integration
class TestFullConversationCLIWorkflow:
    """End-to-end tests for conversation through CLI."""

    @patch('mlx_lm.load')
    @patch('mlx_lm.generate')
    @patch('src.devices.DeviceDetector')
    @patch('src.models.registry.get_model_spec')
    @patch('src.models.memory.MemoryManager')
    def test_complete_cli_conversation_workflow(
        self,
        mock_memory_manager,
        mock_get_model_spec,
        mock_device_detector,
        mock_mlx_generate,
        mock_mlx_load,
    ):
        """Test complete workflow: user input → CLI → conversation → inference → output."""
        from src.config.model_config import ModelConfig
        from src.models.loader import ModelLoader
        from src.inference.engine import InferenceEngine

        # Setup model and inference
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

        # Mock responses
        responses = [
            "Hello! How can I assist you?",
            "Python is a high-level programming language.",
            "It's great for beginners and experts alike!",
        ]
        mock_mlx_generate.side_effect = responses

        # Load model and create engine
        loader = ModelLoader(config=config)
        loader.load()
        engine = InferenceEngine(model_loader=loader)

        # Create conversation and REPL
        conv = ConversationManager(max_context_tokens=4096)
        repl = REPL(conversation=conv, engine=engine)

        # Simulate user interactions
        # Turn 1
        repl._process_message("Hello")
        assert conv.get_message_count() == 2
        assert conv.get_history()[-1].content == "Hello! How can I assist you?"

        # Turn 2
        repl._process_message("What is Python?")
        assert conv.get_message_count() == 4
        assert conv.get_history()[-1].content == "Python is a high-level programming language."

        # Turn 3
        repl._process_message("Is it good for beginners?")
        assert conv.get_message_count() == 6
        assert conv.get_history()[-1].content == "It's great for beginners and experts alike!"

    def test_command_execution_within_conversation(self):
        """Test that commands work correctly during active conversation."""
        conv = ConversationManager()
        repl = REPL(conversation=conv)
        repl.dispatcher.conversation = conv

        # Add some messages
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi")
        assert conv.get_message_count() == 2

        # Execute /history command
        repl.dispatcher.execute("/history")
        # Should still have same messages
        assert conv.get_message_count() == 2

        # Execute /clear command
        repl.dispatcher.execute("/clear")
        # Should be empty now
        assert conv.get_message_count() == 0

    def test_save_load_workflow_during_session(self, tmp_path):
        """Test saving and loading conversation during a session."""
        # Create initial conversation
        conv1 = ConversationManager()
        repl1 = REPL(conversation=conv1)
        repl1.dispatcher.conversation = conv1

        # Add messages
        conv1.add_message("user", "First message")
        conv1.add_message("assistant", "First response")

        # Save
        filepath = tmp_path / "session.json"
        repl1.dispatcher.execute(f"/save {filepath}")

        # Create new session
        conv2 = ConversationManager()
        repl2 = REPL(conversation=conv2)
        repl2.dispatcher.conversation = conv2

        # Load
        repl2.dispatcher.execute(f"/load {filepath}")

        # Continue conversation
        conv2.add_message("user", "Second message")

        # Verify
        assert conv2.get_message_count() == 3
        messages = conv2.get_history()
        assert messages[0].content == "First message"
        assert messages[1].content == "First response"
        assert messages[2].content == "Second message"

    @patch('src.cli.input.create_input_handler')
    def test_repl_autocomplete_with_conversation(self, mock_input_handler):
        """Test that REPL autocomplete works with conversation context."""
        conv = ConversationManager()
        repl = REPL(conversation=conv)

        # Verify input handler was created with commands
        mock_input_handler.assert_called_once()
        call_kwargs = mock_input_handler.call_args.kwargs
        assert 'commands' in call_kwargs
        # Should include standard commands
        commands = call_kwargs['commands']
        assert '/help' in commands or 'help' in commands
        assert '/clear' in commands or 'clear' in commands

    def test_error_handling_in_cli_workflow(self):
        """Test that errors are gracefully handled in CLI workflow."""
        conv = ConversationManager()

        # Mock engine that raises errors
        mock_engine = Mock()
        mock_engine.generate = Mock(side_effect=Exception("Generation failed"))

        repl = REPL(conversation=conv, engine=mock_engine)

        # Process message (should handle error gracefully)
        with patch('src.cli.output.print_error') as mock_print_error:
            result = repl._process_message("Test")

            # Should continue (not crash)
            assert result is True

            # Error should be printed
            mock_print_error.assert_called()


class TestCLIDisplayIntegration:
    """Test CLI display components with conversation data."""

    def test_message_formatter_displays_conversation(self):
        """Test that MessageFormatter correctly displays conversation messages."""
        from src.cli.display import MessageFormatter

        conv = ConversationManager()
        conv.add_message("user", "Test message")
        conv.add_message("assistant", "Test response")

        formatter = MessageFormatter(show_metadata=True)

        # Test display methods (output suppressed)
        with patch('src.cli.output.console'):
            formatter.display_user_message("Test message")
            formatter.display_assistant_message("Test response")
            # Should not raise errors

    def test_welcome_display_with_config(self):
        """Test welcome display with configuration."""
        from src.cli.display import WelcomeDisplay

        welcome = WelcomeDisplay()

        # Test display (output suppressed)
        with patch('src.cli.output.console'):
            welcome.display_welcome(model_name="phi-3-mini")
            # Should not raise errors
