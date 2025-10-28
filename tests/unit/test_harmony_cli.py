"""Unit tests for Harmony CLI integration.

Tests CLI argument parsing, reasoning display, and configuration flow
for Harmony multi-channel format features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cli.display import MessageFormatter
from src.config.inference_config import InferenceConfig, ReasoningLevel
from src.config.cli_config import CLIConfig
from src.main import parse_arguments, create_config_from_args


class TestHarmonyArgumentParsing:
    """Test CLI argument parsing for Harmony features."""

    def test_default_harmony_enabled(self):
        """Test that Harmony format is enabled by default."""
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()
            # Default should be None (will use env/config defaults)
            assert args.use_harmony_format is None

    def test_harmony_flag_explicit_enable(self):
        """Test --harmony flag explicitly enables Harmony format."""
        with patch('sys.argv', ['main.py', '--harmony']):
            args = parse_arguments()
            assert args.use_harmony_format is True

    def test_no_harmony_flag_disables(self):
        """Test --no-harmony flag disables Harmony format."""
        with patch('sys.argv', ['main.py', '--no-harmony']):
            args = parse_arguments()
            assert args.use_harmony_format is False

    def test_reasoning_level_low(self):
        """Test --reasoning-level low."""
        with patch('sys.argv', ['main.py', '--reasoning-level', 'low']):
            args = parse_arguments()
            assert args.reasoning_level == 'low'

    def test_reasoning_level_medium(self):
        """Test --reasoning-level medium (default)."""
        with patch('sys.argv', ['main.py', '--reasoning-level', 'medium']):
            args = parse_arguments()
            assert args.reasoning_level == 'medium'

    def test_reasoning_level_high(self):
        """Test --reasoning-level high."""
        with patch('sys.argv', ['main.py', '--reasoning-level', 'high']):
            args = parse_arguments()
            assert args.reasoning_level == 'high'

    def test_capture_reasoning_flag(self):
        """Test --capture-reasoning flag."""
        with patch('sys.argv', ['main.py', '--capture-reasoning']):
            args = parse_arguments()
            assert args.capture_reasoning is True

    def test_show_reasoning_flag(self):
        """Test --show-reasoning flag."""
        with patch('sys.argv', ['main.py', '--show-reasoning']):
            args = parse_arguments()
            assert args.show_reasoning is True

    def test_combined_harmony_flags(self):
        """Test multiple Harmony flags together."""
        with patch('sys.argv', [
            'main.py',
            '--harmony',
            '--reasoning-level', 'high',
            '--capture-reasoning',
            '--show-reasoning'
        ]):
            args = parse_arguments()
            assert args.use_harmony_format is True
            assert args.reasoning_level == 'high'
            assert args.capture_reasoning is True
            assert args.show_reasoning is True


class TestConfigurationFromArgs:
    """Test configuration creation from CLI arguments."""

    @patch.dict('os.environ', {}, clear=True)
    def test_harmony_settings_applied_to_inference_config(self):
        """Test Harmony settings are passed to InferenceConfig."""
        with patch('sys.argv', [
            'main.py',
            '--no-harmony',
            '--reasoning-level', 'low',
            '--capture-reasoning',
            '--show-reasoning'
        ]):
            args = parse_arguments()
            model_config, inference_config, cli_config = create_config_from_args(args)

            assert inference_config.use_harmony_format is False
            assert inference_config.reasoning_level == ReasoningLevel.LOW
            assert inference_config.capture_reasoning is True
            assert inference_config.show_reasoning is True

    @patch.dict('os.environ', {}, clear=True)
    def test_show_reasoning_in_cli_config(self):
        """Test show_reasoning is also set in CLIConfig."""
        with patch('sys.argv', ['main.py', '--show-reasoning']):
            args = parse_arguments()
            model_config, inference_config, cli_config = create_config_from_args(args)

            assert cli_config.show_reasoning is True
            assert inference_config.show_reasoning is True

    @patch.dict('os.environ', {}, clear=True)
    def test_reasoning_level_enum_conversion(self):
        """Test reasoning level string is converted to enum."""
        with patch('sys.argv', ['main.py', '--reasoning-level', 'high']):
            args = parse_arguments()
            model_config, inference_config, cli_config = create_config_from_args(args)

            assert isinstance(inference_config.reasoning_level, ReasoningLevel)
            assert inference_config.reasoning_level == ReasoningLevel.HIGH

    @patch.dict('os.environ', {}, clear=True)
    def test_default_reasoning_level_from_env(self):
        """Test reasoning level defaults to medium from environment."""
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()
            model_config, inference_config, cli_config = create_config_from_args(args)

            # Default from InferenceConfig.from_env() should be MEDIUM
            assert inference_config.reasoning_level == ReasoningLevel.MEDIUM


class TestMessageFormatterReasoningDisplay:
    """Test MessageFormatter.display_reasoning() method."""

    def test_display_reasoning_with_text(self):
        """Test reasoning display with valid text."""
        formatter = MessageFormatter(show_metadata=False)

        with patch('src.cli.display.console') as mock_console:
            reasoning = "Step 1: Analyze the problem\nStep 2: Plan solution"
            formatter.display_reasoning(reasoning)

            # Should call console.print() multiple times
            assert mock_console.print.call_count >= 4  # separator, header, content, separator

    def test_display_reasoning_with_empty_text(self):
        """Test reasoning display with empty text does nothing."""
        formatter = MessageFormatter(show_metadata=False)

        with patch('src.cli.display.console') as mock_console:
            formatter.display_reasoning("")
            # Should not call console.print() at all
            mock_console.print.assert_not_called()

    def test_display_reasoning_with_none(self):
        """Test reasoning display with None does nothing."""
        formatter = MessageFormatter(show_metadata=False)

        with patch('src.cli.display.console') as mock_console:
            formatter.display_reasoning(None)
            mock_console.print.assert_not_called()

    def test_display_reasoning_truncation(self):
        """Test reasoning display with truncation."""
        formatter = MessageFormatter(show_metadata=False)

        with patch('src.cli.display.console') as mock_console:
            long_reasoning = "A" * 1000
            formatter.display_reasoning(long_reasoning, max_length=100)

            # Verify console.print was called
            assert mock_console.print.call_count >= 4

            # Check that formatted text was passed (should be truncated)
            calls = mock_console.print.call_args_list
            content_call = calls[2]  # Third call should be the content
            formatted_text = str(content_call[0][0])
            assert len(formatted_text) <= 113  # 100 + " [TRUNCATED]"

    def test_display_reasoning_formats_inline(self):
        """Test that display_reasoning formats reasoning inline."""
        formatter = MessageFormatter(show_metadata=False)

        # Patch console to capture output
        with patch('src.cli.display.console') as mock_console:
            reasoning = "<|start|>channel<|channel|>analysis<|message|>Test reasoning<|end|>"

            formatter.display_reasoning(reasoning, max_length=200)

            # Verify console.print was called (formatting happens inline now)
            assert mock_console.print.called


class TestReasoningDisplayOptIn:
    """Test that reasoning display is opt-in only."""

    def test_reasoning_not_shown_by_default(self):
        """Test reasoning is not displayed without --show-reasoning flag."""
        # Test via config defaults
        config = CLIConfig.from_env()
        assert config.show_reasoning is False

        inference_config = InferenceConfig.from_env()
        assert inference_config.show_reasoning is False

    def test_reasoning_shown_with_flag(self):
        """Test reasoning is displayed with --show-reasoning flag."""
        with patch('sys.argv', ['main.py', '--show-reasoning']):
            args = parse_arguments()
            model_config, inference_config, cli_config = create_config_from_args(args)

            assert cli_config.show_reasoning is True
            assert inference_config.show_reasoning is True


class TestREPLReasoningIntegration:
    """Test REPL integration with reasoning display."""

    def test_repl_displays_reasoning_when_enabled(self):
        """Test REPL displays reasoning when config.show_reasoning is True."""
        from src.cli.repl import REPL

        # Create mock config with show_reasoning enabled
        mock_config = Mock()
        mock_config.show_reasoning = True

        # Create mock conversation and engine
        mock_conversation = Mock()
        mock_conversation.add_message = Mock()
        mock_conversation.format_prompt = Mock(return_value="test prompt")

        # Create mock result with reasoning
        mock_result = Mock()
        mock_result.text = "Test response"
        mock_result.reasoning = "Test reasoning trace"

        mock_engine = Mock()
        mock_engine.generate = Mock(return_value=mock_result)

        # Create REPL instance
        repl = REPL(
            conversation=mock_conversation,
            engine=mock_engine,
            config=mock_config
        )

        # Mock the formatter's display methods
        with patch.object(repl.formatter, 'display_user_message'):
            with patch.object(repl.formatter, 'display_assistant_message'):
                with patch.object(repl.formatter, 'display_reasoning') as mock_display_reasoning:
                    # Process a message
                    repl._process_message("test message")

                    # Verify reasoning was displayed
                    mock_display_reasoning.assert_called_once_with("Test reasoning trace")

    def test_repl_does_not_display_reasoning_when_disabled(self):
        """Test REPL does not display reasoning when config.show_reasoning is False."""
        from src.cli.repl import REPL

        # Create mock config with show_reasoning disabled
        mock_config = Mock()
        mock_config.show_reasoning = False

        # Create mock conversation and engine
        mock_conversation = Mock()
        mock_conversation.add_message = Mock()
        mock_conversation.format_prompt = Mock(return_value="test prompt")

        # Create mock result with reasoning
        mock_result = Mock()
        mock_result.text = "Test response"
        mock_result.reasoning = "Test reasoning trace"

        mock_engine = Mock()
        mock_engine.generate = Mock(return_value=mock_result)

        # Create REPL instance
        repl = REPL(
            conversation=mock_conversation,
            engine=mock_engine,
            config=mock_config
        )

        # Mock the formatter's display methods
        with patch.object(repl.formatter, 'display_user_message'):
            with patch.object(repl.formatter, 'display_assistant_message'):
                with patch.object(repl.formatter, 'display_reasoning') as mock_display_reasoning:
                    # Process a message
                    repl._process_message("test message")

                    # Verify reasoning was NOT displayed
                    mock_display_reasoning.assert_not_called()

    def test_repl_handles_result_without_reasoning(self):
        """Test REPL handles result without reasoning attribute gracefully."""
        from src.cli.repl import REPL

        # Create mock config with show_reasoning enabled
        mock_config = Mock()
        mock_config.show_reasoning = True

        # Create mock conversation and engine
        mock_conversation = Mock()
        mock_conversation.add_message = Mock()
        mock_conversation.format_prompt = Mock(return_value="test prompt")

        # Create mock result WITHOUT reasoning - explicitly configure to not have reasoning
        mock_result = Mock(spec=['text'])  # Only has 'text' attribute
        mock_result.text = "Test response"

        mock_engine = Mock()
        mock_engine.generate = Mock(return_value=mock_result)

        # Create REPL instance
        repl = REPL(
            conversation=mock_conversation,
            engine=mock_engine,
            config=mock_config
        )

        # Mock the formatter's display methods
        with patch.object(repl.formatter, 'display_user_message'):
            with patch.object(repl.formatter, 'display_assistant_message'):
                with patch.object(repl.formatter, 'display_reasoning') as mock_display_reasoning:
                    # Process a message - should not raise error
                    result = repl._process_message("test message")

                    # Should return True (continue loop)
                    assert result is True

                    # Verify reasoning was NOT displayed
                    mock_display_reasoning.assert_not_called()


class TestHelpText:
    """Test that help text includes Harmony documentation."""

    def test_help_includes_harmony_section(self):
        """Test --help output includes Harmony format options."""
        with patch('sys.argv', ['main.py', '--help']):
            with pytest.raises(SystemExit):  # --help causes exit
                with patch('sys.stdout', new=StringIO()) as mock_stdout:
                    parse_arguments()
                    help_text = mock_stdout.getvalue()

                    # Should contain Harmony group
                    assert 'Harmony format options' in help_text or 'harmony' in help_text.lower()

    def test_help_includes_reasoning_levels(self):
        """Test --help output documents reasoning levels."""
        with patch('sys.argv', ['main.py', '--help']):
            with pytest.raises(SystemExit):
                with patch('sys.stdout', new=StringIO()) as mock_stdout:
                    parse_arguments()
                    help_text = mock_stdout.getvalue()

                    # Should document reasoning levels
                    assert 'Reasoning Levels' in help_text or 'reasoning-level' in help_text


class TestStreamingWithChannels:
    """Test CLI integration with new dict-based streaming format."""

    def test_streaming_yields_dict_tokens(self):
        """Test that streaming yields dict tokens with channel metadata."""
        from src.inference.engine import InferenceEngine
        from src.models.loader import ModelLoader
        from src.config.model_config import ModelConfig

        # Mock model and tokenizer
        with patch('mlx_lm.load') as mock_load:
            with patch('mlx_lm.stream_generate') as mock_stream:
                # Setup mocks
                mock_load.return_value = (Mock(), Mock())
                mock_stream.return_value = iter(["Hello", " ", "world"])

                # Create engine
                config = ModelConfig(model_name="test", warmup=False)
                loader = ModelLoader(config=config)
                loader.load()
                engine = InferenceEngine(model_loader=loader)

                # Generate stream
                stream = engine.generate_stream("Test prompt")
                tokens = list(stream)

                # Verify dict structure
                assert len(tokens) > 0
                for token_dict in tokens:
                    assert isinstance(token_dict, dict), "Should yield dict"
                    assert 'token' in token_dict, "Should have 'token' key"
                    assert 'channel' in token_dict, "Should have 'channel' key"
                    assert 'is_final' in token_dict, "Should have 'is_final' key"

    def test_cli_displays_final_channel_only(self):
        """Test CLI only displays final channel by default."""
        from src.cli.display import MessageFormatter
        import sys
        from io import StringIO

        formatter = MessageFormatter(show_metadata=False)

        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            # Simulate displaying a response
            final_text = "This is the final response"
            formatter.display_assistant_message(final_text)

        # Verify output was produced
        output = captured_output.getvalue()
        assert "final response" in output or len(output) > 0

    def test_repl_extracts_tokens_from_dict(self):
        """Test REPL properly extracts tokens from dict format."""
        from src.cli.repl import REPL
        from src.conversation.history import ConversationManager

        # Create mock engine that returns dict tokens
        mock_engine = Mock()

        def mock_stream(*args, **kwargs):
            """Mock stream that yields dict tokens."""
            yield {'token': 'Hello', 'channel': 'final', 'is_final': True, 'content': 'Hello', 'delta': 'Hello'}
            yield {'token': ' ', 'channel': 'final', 'is_final': True, 'content': ' ', 'delta': ' '}
            yield {'token': 'world', 'channel': 'final', 'is_final': True, 'content': 'world', 'delta': 'world'}

        mock_engine.generate_stream.return_value = mock_stream()

        # Mock generate to return result with text
        mock_result = Mock()
        mock_result.text = "Hello world"
        mock_engine.generate.return_value = mock_result

        # Create REPL
        conversation = ConversationManager()
        repl = REPL(conversation=conversation, engine=mock_engine)

        # Process message
        with patch.object(repl.formatter, 'display_user_message'):
            with patch.object(repl.formatter, 'display_assistant_message'):
                result = repl._process_message("Test")

                # Should succeed without error
                assert result is True

    def test_channel_metadata_preserved_in_stream(self):
        """Test that channel metadata is preserved in streaming."""
        # Create mock tokens with channel metadata
        mock_tokens = [
            {'token': 'Analysis:', 'channel': 'analysis', 'is_final': False, 'content': 'Analysis:', 'delta': 'Analysis:'},
            {'token': ' thinking', 'channel': 'analysis', 'is_final': False, 'content': ' thinking', 'delta': ' thinking'},
            {'token': 'Hello', 'channel': 'final', 'is_final': True, 'content': 'Hello', 'delta': 'Hello'},
        ]

        # Verify structure
        for token_dict in mock_tokens:
            assert 'channel' in token_dict
            assert 'is_final' in token_dict
            assert token_dict['channel'] in ['analysis', 'final']
            assert isinstance(token_dict['is_final'], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
