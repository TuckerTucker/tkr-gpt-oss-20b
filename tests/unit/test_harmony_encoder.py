"""
Unit tests for Harmony Encoder.

Tests the HarmonyEncoder class implementation including:
- Conversation encoding with various message types
- Response parsing with single and multiple channels
- Format validation
- Error handling for malformed input
- Fallback parser behavior
- Performance benchmarks
"""

import pytest
import sys
import os
from unittest.mock import patch
import time

# Import contract types
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-integration', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_encoder_interface import (
    Role,
    Channel,
    HarmonyMessage,
    ParsedResponse,
)

from src.prompts.harmony import HarmonyEncoder


class TestHarmonyEncoderBasics:
    """Test basic encoding and initialization."""

    def test_encoder_initialization(self):
        """Test that encoder initializes correctly."""
        encoder = HarmonyEncoder()
        assert encoder is not None
        assert hasattr(encoder, 'encode_conversation')
        assert hasattr(encoder, 'parse_response')
        assert hasattr(encoder, 'validate_format')

    def test_encode_simple_conversation(self):
        """Test encoding a simple user message."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.USER, content="Hello!")
        ]

        result = encoder.encode_conversation(messages)

        assert "<|start|>user" in result
        assert "<|message|>Hello!" in result
        assert "<|end|>" in result

    def test_encode_with_system_message(self):
        """Test encoding conversation with system message."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.SYSTEM, content="You are helpful."),
            HarmonyMessage(role=Role.USER, content="Hello!")
        ]

        result = encoder.encode_conversation(messages)

        assert "<|start|>system" in result
        assert "<|message|>You are helpful." in result
        assert "<|start|>user" in result
        assert "<|message|>Hello!" in result

    def test_encode_with_developer_role(self):
        """Test encoding with developer role message."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.DEVELOPER, content="Debug mode enabled"),
            HarmonyMessage(role=Role.USER, content="Test")
        ]

        result = encoder.encode_conversation(messages)

        assert "<|start|>developer" in result
        assert "<|message|>Debug mode enabled" in result

    def test_encode_assistant_with_channel(self):
        """Test encoding assistant message with channel."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(
                role=Role.ASSISTANT,
                content="I'm thinking...",
                channel=Channel.ANALYSIS
            )
        ]

        result = encoder.encode_conversation(messages)

        assert "<|start|>assistant" in result
        assert "<|channel|>analysis" in result
        assert "<|message|>I'm thinking..." in result

    def test_encode_with_generation_prompt(self):
        """Test that generation prompt is added when requested."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.USER, content="Hello!")
        ]

        result = encoder.encode_conversation(messages, include_generation_prompt=True)

        # Should end with assistant start token
        assert result.endswith("<|start|>assistant")

    def test_encode_without_generation_prompt(self):
        """Test encoding without generation prompt."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.USER, content="Hello!")
        ]

        result = encoder.encode_conversation(messages, include_generation_prompt=False)

        # Should end with end token, not assistant start
        assert result.endswith("<|end|>")
        assert not result.endswith("<|start|>assistant")


class TestHarmonyMessageValidation:
    """Test message validation rules."""

    def test_assistant_message_requires_channel(self):
        """Test that assistant messages must have a channel."""
        # This should raise when validated during encoding
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.ASSISTANT, content="Hello")
            # Missing channel - should fail validation
        ]

        with pytest.raises(ValueError, match="must specify a channel"):
            encoder.encode_conversation(messages)

    def test_non_assistant_cannot_have_channel(self):
        """Test that non-assistant messages cannot have channels."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(
                role=Role.USER,
                content="Hello",
                channel=Channel.FINAL  # Invalid for user role
            )
        ]

        with pytest.raises(ValueError, match="Only assistant messages"):
            encoder.encode_conversation(messages)


class TestResponseParsing:
    """Test parsing of model responses."""

    def test_parse_single_channel_final(self):
        """Test parsing response with only final channel."""
        encoder = HarmonyEncoder()
        response = "<|start|>assistant<|channel|>final<|message|>Hello there!<|end|>"

        parsed = encoder.parse_response(response)

        assert parsed.final == "Hello there!"
        assert parsed.analysis is None
        assert parsed.commentary is None
        assert parsed.raw == response

    def test_parse_multiple_channels(self):
        """Test parsing response with multiple channels."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>analysis<|message|>Thinking step by step<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>Executing function<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Here's the answer<|end|>"
        )

        parsed = encoder.parse_response(response)

        assert parsed.final == "Here's the answer"
        assert parsed.analysis == "Thinking step by step"
        assert parsed.commentary == "Executing function"

    def test_parse_extract_final_only(self):
        """Test parsing with extract_final_only=True."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>analysis<|message|>Internal reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>User response<|end|>"
        )

        parsed = encoder.parse_response(response, extract_final_only=True)

        assert parsed.final == "User response"
        # Analysis should not be extracted when extract_final_only=True
        assert parsed.analysis is None

    def test_parse_whitespace_handling(self):
        """Test that whitespace is properly trimmed."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>final<|message|>  \n"
            "  Response with whitespace  \n"
            "  <|end|>"
        )

        parsed = encoder.parse_response(response)

        # Should strip leading/trailing whitespace but preserve internal
        assert parsed.final == "Response with whitespace"

    def test_parse_multiline_content(self):
        """Test parsing content with multiple lines."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Line 1\nLine 2\nLine 3"
            "<|end|>"
        )

        parsed = encoder.parse_response(response)

        assert "Line 1\nLine 2\nLine 3" in parsed.final


class TestMalformedInput:
    """Test error handling for malformed input."""

    def test_parse_missing_harmony_markers(self):
        """Test parsing text without Harmony markers."""
        encoder = HarmonyEncoder()
        response = "Just plain text without markers"

        # Should fall back to using plain text as final
        parsed = encoder.parse_response(response)
        assert parsed.final == "Just plain text without markers"

    def test_parse_incomplete_message(self):
        """Test parsing incomplete message structure."""
        encoder = HarmonyEncoder()
        response = "<|start|>assistant<|channel|>final"
        # Missing message token and end token

        # Should handle gracefully or raise error
        parsed = encoder.parse_response(response)
        # Should fall back to cleaned text
        assert "assistant" in parsed.final or parsed.final == ""

    def test_parse_empty_response(self):
        """Test parsing empty response."""
        encoder = HarmonyEncoder()
        response = ""

        with pytest.raises(ValueError):
            encoder.parse_response(response)

    def test_encode_empty_messages_list(self):
        """Test encoding with empty messages list."""
        encoder = HarmonyEncoder()
        messages = []

        result = encoder.encode_conversation(messages)

        # Should return generation prompt only if requested
        assert result == "<|start|>assistant" or result == ""


class TestFormatValidation:
    """Test Harmony format validation."""

    def test_validate_valid_format(self):
        """Test validation of properly formatted Harmony text."""
        encoder = HarmonyEncoder()
        text = (
            "<|start|>user<|message|>Hello<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hi<|end|>"
        )

        assert encoder.validate_format(text) is True

    def test_validate_missing_end_token(self):
        """Test validation fails for missing end token."""
        encoder = HarmonyEncoder()
        text = "<|start|>user<|message|>Hello"  # Missing <|end|>

        assert encoder.validate_format(text) is False

    def test_validate_unbalanced_tokens(self):
        """Test validation fails for unbalanced start/end tokens."""
        encoder = HarmonyEncoder()
        text = (
            "<|start|>user<|message|>Hello<|end|>"
            "<|start|>assistant<|message|>Hi"  # Missing end
        )

        assert encoder.validate_format(text) is False

    def test_validate_invalid_role(self):
        """Test validation fails for invalid role."""
        encoder = HarmonyEncoder()
        text = "<|start|>invalid_role<|message|>Hello<|end|>"

        assert encoder.validate_format(text) is False

    def test_validate_empty_string(self):
        """Test validation fails for empty string."""
        encoder = HarmonyEncoder()

        assert encoder.validate_format("") is False

    def test_validate_no_tokens(self):
        """Test validation fails for text without tokens."""
        encoder = HarmonyEncoder()
        text = "Just plain text"

        assert encoder.validate_format(text) is False

    def test_validate_missing_message_token(self):
        """Test validation fails without message token."""
        encoder = HarmonyEncoder()
        text = "<|start|>user<|end|>"  # Missing <|message|>

        assert encoder.validate_format(text) is False


class TestFallbackParser:
    """Test fallback regex-based parser behavior."""

    def test_fallback_parser_directly(self):
        """Test the regex fallback parser directly."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>final<|message|>Fallback test<|end|>"
        )

        # Call fallback parser directly
        parsed = encoder._parse_with_regex(response, extract_final_only=False)

        assert parsed.final == "Fallback test"

    def test_fallback_with_harmony_unavailable(self):
        """Test behavior when openai-harmony is unavailable."""
        # Create encoder and manually set _harmony_available to False
        encoder = HarmonyEncoder()
        encoder._harmony_available = False

        response = "<|start|>assistant<|channel|>final<|message|>Test<|end|>"
        parsed = encoder.parse_response(response)

        assert parsed.final == "Test"


class TestPerformance:
    """Test performance requirements."""

    def test_encode_performance_20_messages(self):
        """Test encoding 20 messages completes in <10ms."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(
                role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                content=f"Message {i}",
                channel=Channel.FINAL if i % 2 == 1 else None
            )
            for i in range(20)
        ]

        start_time = time.perf_counter()
        result = encoder.encode_conversation(messages)
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed < 10, f"Encoding took {elapsed:.2f}ms, expected <10ms"
        assert len(result) > 0

    def test_parse_performance_1kb_response(self):
        """Test parsing 1KB response completes in <5ms."""
        encoder = HarmonyEncoder()

        # Create ~1KB response
        content = "x" * 900  # ~900 bytes of content
        response = (
            f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
        )

        assert len(response) >= 1000  # Verify we have at least 1KB

        start_time = time.perf_counter()
        parsed = encoder.parse_response(response)
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms

        assert elapsed < 5, f"Parsing took {elapsed:.2f}ms, expected <5ms"
        assert len(parsed.final) > 0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_encode_special_characters(self):
        """Test encoding messages with special characters."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(
                role=Role.USER,
                content="Special chars: <>&\"'{}[]|\n\t"
            )
        ]

        result = encoder.encode_conversation(messages)

        assert "Special chars: <>&\"'{}[]|" in result

    def test_parse_multiple_final_channels(self):
        """Test parsing response with multiple final channels (last wins)."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>final<|message|>First final<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Second final<|end|>"
        )

        parsed = encoder.parse_response(response)

        # The last final channel should be used
        assert parsed.final == "Second final"

    def test_parse_channel_order_independence(self):
        """Test that channel order doesn't matter."""
        encoder = HarmonyEncoder()
        response = (
            "<|start|>assistant<|channel|>final<|message|>Final text<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>Analysis text<|end|>"
        )

        parsed = encoder.parse_response(response)

        assert parsed.final == "Final text"
        assert parsed.analysis == "Analysis text"

    def test_long_conversation_encoding(self):
        """Test encoding a very long conversation."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(
                role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
                content=f"Message number {i} with some content",
                channel=Channel.FINAL if i % 2 == 1 else None
            )
            for i in range(100)
        ]

        result = encoder.encode_conversation(messages)

        # Verify structure is maintained
        assert result.count("<|start|>") == 101  # 100 messages + generation prompt
        assert result.count("<|end|>") == 100

    def test_unicode_content(self):
        """Test encoding and parsing with Unicode content."""
        encoder = HarmonyEncoder()
        messages = [
            HarmonyMessage(role=Role.USER, content="Hello 世界 🌍"),
        ]

        encoded = encoder.encode_conversation(messages)
        assert "Hello 世界 🌍" in encoded

        response = "<|start|>assistant<|channel|>final<|message|>Привет мир 🚀<|end|>"
        parsed = encoder.parse_response(response)
        assert parsed.final == "Привет мир 🚀"


class TestThreadSafety:
    """Test thread safety requirements."""

    def test_concurrent_encoding(self):
        """Test that concurrent encoding operations work correctly."""
        import threading

        encoder = HarmonyEncoder()
        results = []
        errors = []

        def encode_task(task_id):
            try:
                messages = [
                    HarmonyMessage(
                        role=Role.USER,
                        content=f"Task {task_id}"
                    )
                ]
                result = encoder.encode_conversation(messages)
                results.append((task_id, result))
            except Exception as e:
                errors.append((task_id, e))

        # Run 10 concurrent encoding tasks
        threads = [threading.Thread(target=encode_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All tasks should complete without errors
        assert len(errors) == 0
        assert len(results) == 10

        # Each result should be unique and valid
        for task_id, result in results:
            assert f"Task {task_id}" in result

    def test_concurrent_parsing(self):
        """Test that concurrent parsing operations work correctly."""
        import threading

        encoder = HarmonyEncoder()
        results = []
        errors = []

        def parse_task(task_id):
            try:
                response = (
                    f"<|start|>assistant<|channel|>final<|message|>"
                    f"Response {task_id}<|end|>"
                )
                parsed = encoder.parse_response(response)
                results.append((task_id, parsed))
            except Exception as e:
                errors.append((task_id, e))

        # Run 10 concurrent parsing tasks
        threads = [threading.Thread(target=parse_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All tasks should complete without errors
        assert len(errors) == 0
        assert len(results) == 10

        # Each result should be unique and valid
        for task_id, parsed in results:
            assert parsed.final == f"Response {task_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/prompts/harmony", "--cov-report=term-missing"])
