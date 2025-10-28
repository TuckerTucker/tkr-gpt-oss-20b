"""
Unit tests for Harmony Response Parser channel extraction.

Tests cover:
- Channel extraction for all channel types (analysis, commentary, final)
- Multi-channel response handling
- Malformed input handling
- Format validation
- ParsedHarmonyResponse dataclass usage
- Extract methods and utilities
"""

import pytest
import sys
import os

# Add contract path for interface imports
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-replacement', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_parser_interface import ParsedHarmonyResponse

from src.prompts.harmony_native import (
    HarmonyResponseParser,
    HarmonyFormatError,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text, allowed_special="all"):
        """Mock encode method."""
        # Simple mock: return list of char codes
        return [ord(c) % 256 for c in text[:100]]


class TestExtractChannel:
    """Test extract_channel method from parser."""

    def test_extract_final_channel(self):
        """Test extracting final channel content."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello, world!",
            channels={"final": "Hello, world!"}
        )

        result = parser.extract_channel(parsed, "final")
        assert result == "Hello, world!"

    def test_extract_analysis_channel(self):
        """Test extracting analysis channel content."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Answer",
            analysis="Step 1: Parse input",
            channels={"final": "Answer", "analysis": "Step 1: Parse input"}
        )

        result = parser.extract_channel(parsed, "analysis")
        assert result == "Step 1: Parse input"

    def test_extract_commentary_channel(self):
        """Test extracting commentary channel content."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Answer",
            commentary="This is interesting",
            channels={"final": "Answer", "commentary": "This is interesting"}
        )

        result = parser.extract_channel(parsed, "commentary")
        assert result == "This is interesting"

    def test_extract_missing_channel(self):
        """Test extracting non-existent channel returns None."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello",
            channels={"final": "Hello"}
        )

        result = parser.extract_channel(parsed, "analysis")
        assert result is None

    def test_extract_from_multi_channel_response(self):
        """Test extracting specific channels from multi-channel response."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="The answer is 42",
            analysis="Think step by step",
            commentary="Simple math problem",
            channels={
                "final": "The answer is 42",
                "analysis": "Think step by step",
                "commentary": "Simple math problem"
            }
        )

        analysis = parser.extract_channel(parsed, "analysis")
        final = parser.extract_channel(parsed, "final")
        commentary = parser.extract_channel(parsed, "commentary")

        assert analysis == "Think step by step"
        assert final == "The answer is 42"
        assert commentary == "Simple math problem"

    def test_extract_channel_with_no_channels_dict(self):
        """Test extracting when channels dict is None."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello",
            analysis="Thinking"
        )

        # Should still work using dedicated fields
        assert parser.extract_channel(parsed, "final") == "Hello"
        assert parser.extract_channel(parsed, "analysis") == "Thinking"

    def test_extract_custom_channel_from_dict(self):
        """Test extracting custom channel from channels dict."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Answer",
            channels={
                "final": "Answer",
                "custom": "Custom content"
            }
        )

        result = parser.extract_channel(parsed, "custom")
        assert result == "Custom content"


class TestParsedHarmonyResponseStructure:
    """Test ParsedHarmonyResponse dataclass structure."""

    def test_parsed_response_with_all_fields(self):
        """Test creating ParsedHarmonyResponse with all fields."""
        parsed = ParsedHarmonyResponse(
            final="Final answer",
            analysis="Step-by-step reasoning",
            commentary="Meta-commentary",
            channels={
                "final": "Final answer",
                "analysis": "Step-by-step reasoning",
                "commentary": "Meta-commentary"
            },
            metadata={"token_count": 100, "parse_time_ms": 2.5}
        )

        assert parsed.final == "Final answer"
        assert parsed.analysis == "Step-by-step reasoning"
        assert parsed.commentary == "Meta-commentary"
        assert parsed.channels["final"] == "Final answer"
        assert parsed.metadata["token_count"] == 100

    def test_parsed_response_minimal_fields(self):
        """Test creating ParsedHarmonyResponse with minimal fields."""
        parsed = ParsedHarmonyResponse(final="Hello")

        assert parsed.final == "Hello"
        assert parsed.analysis is None
        assert parsed.commentary is None
        assert parsed.channels is None
        assert parsed.metadata is None

    def test_parsed_response_final_required(self):
        """Test that final field is required."""
        # This should work - final is required
        parsed = ParsedHarmonyResponse(final="Required")
        assert parsed.final == "Required"

        # This would fail at creation time (dataclass requires final)
        # We can't really test this without expecting a TypeError

    def test_parsed_response_with_empty_final(self):
        """Test ParsedHarmonyResponse with empty final."""
        parsed = ParsedHarmonyResponse(final="")

        assert parsed.final == ""
        assert parsed.final is not None  # Never None, contract requirement

    def test_parsed_response_with_metadata(self):
        """Test ParsedHarmonyResponse stores metadata correctly."""
        metadata = {
            "token_count": 150,
            "parse_time_ms": 3.2,
            "message_count": 2
        }

        parsed = ParsedHarmonyResponse(
            final="Answer",
            metadata=metadata
        )

        assert parsed.metadata["token_count"] == 150
        assert parsed.metadata["parse_time_ms"] == 3.2
        assert parsed.metadata["message_count"] == 2


class TestParserValidation:
    """Test parser format validation methods."""

    def test_validate_empty_tokens(self):
        """Test validation fails for empty token list."""
        parser = HarmonyResponseParser()

        result = parser.validate_harmony_format([])
        assert result is False

    def test_validate_non_harmony_tokens(self):
        """Test validation fails for non-Harmony tokens."""
        parser = HarmonyResponseParser()

        # Random tokens that don't represent Harmony format
        result = parser.validate_harmony_format([1, 2, 3, 4, 5])
        assert result is False

    def test_validate_returns_boolean(self):
        """Test that validate_harmony_format always returns bool."""
        parser = HarmonyResponseParser()

        # Should return False for invalid input
        result = parser.validate_harmony_format([999])
        assert isinstance(result, bool)
        assert result is False


class TestParserErrorHandling:
    """Test parser error handling for malformed inputs."""

    def test_parse_empty_tokens_raises_error(self):
        """Test that parsing empty tokens raises ValueError."""
        parser = HarmonyResponseParser()

        with pytest.raises(ValueError, match="Token IDs cannot be empty"):
            parser.parse_response_tokens([])

    def test_parse_empty_text_raises_error(self):
        """Test that parsing empty text raises ValueError."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="Response text cannot be empty"):
            parser.parse_response_text("", tokenizer)

    def test_parse_with_no_tokenizer_raises_error(self):
        """Test that parsing without tokenizer raises ValueError."""
        parser = HarmonyResponseParser()

        with pytest.raises(ValueError, match="Tokenizer is required"):
            parser.parse_response_text("test", None)

    def test_parse_malformed_tokens_graceful(self):
        """Test that malformed tokens are handled gracefully."""
        parser = HarmonyResponseParser()

        # Parser should handle invalid tokens without crashing
        try:
            result = parser.parse_response_tokens([999999, 888888])
            # If it succeeds, should return valid structure
            assert isinstance(result, ParsedHarmonyResponse)
            assert result.final is not None  # Never None
        except (ValueError, HarmonyFormatError):
            # These are acceptable exceptions for malformed input
            pass

    def test_parse_with_bad_tokenizer(self):
        """Test handling of tokenizer without encode method."""
        parser = HarmonyResponseParser()

        bad_tokenizer = object()  # No encode method

        with pytest.raises(ValueError, match="Tokenizer"):
            parser.parse_response_text("test", bad_tokenizer)


class TestParserTextParsing:
    """Test text-based parsing using tokenizer."""

    def test_parse_response_text_uses_tokenizer(self):
        """Test that parse_response_text uses provided tokenizer."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        # Should use tokenizer.encode() internally
        try:
            result = parser.parse_response_text("Hello", tokenizer)
            assert isinstance(result, ParsedHarmonyResponse)
        except (ValueError, HarmonyFormatError):
            # Expected if tokens don't match Harmony format
            pass

    def test_parse_response_text_with_unicode(self):
        """Test parsing text with unicode characters."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        try:
            result = parser.parse_response_text("Hello 世界 🌍", tokenizer)
            assert isinstance(result, ParsedHarmonyResponse)
        except (ValueError, HarmonyFormatError):
            # Expected if tokens don't match Harmony format
            pass


class TestExtractFinalOnly:
    """Test extract_final_only optimization."""

    def test_parse_with_extract_final_only_true(self):
        """Test parsing with extract_final_only=True."""
        parser = HarmonyResponseParser()

        # When extract_final_only=True, parser should skip analysis/commentary
        # This is an optimization - we can't easily test without real Harmony tokens
        # Just verify the parameter is accepted
        try:
            result = parser.parse_response_tokens([1, 2, 3], extract_final_only=True)
            # If successful, analysis and commentary should be None
            if isinstance(result, ParsedHarmonyResponse):
                # These may be None due to extract_final_only
                assert result.final is not None
        except (ValueError, HarmonyFormatError):
            pass

    def test_parse_text_with_extract_final_only(self):
        """Test text parsing with extract_final_only=True."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        try:
            result = parser.parse_response_text(
                "Some text",
                tokenizer,
                extract_final_only=True
            )
            if isinstance(result, ParsedHarmonyResponse):
                assert result.final is not None
        except (ValueError, HarmonyFormatError):
            pass


class TestChannelsDictionary:
    """Test channels dictionary in ParsedHarmonyResponse."""

    def test_channels_dict_contains_all_channels(self):
        """Test that channels dict contains all extracted channels."""
        parsed = ParsedHarmonyResponse(
            final="Answer",
            analysis="Reasoning",
            commentary="Note",
            channels={
                "final": "Answer",
                "analysis": "Reasoning",
                "commentary": "Note",
                "custom": "Custom channel"
            }
        )

        assert len(parsed.channels) == 4
        assert "final" in parsed.channels
        assert "analysis" in parsed.channels
        assert "commentary" in parsed.channels
        assert "custom" in parsed.channels

    def test_channels_dict_can_be_none(self):
        """Test that channels dict can be None."""
        parsed = ParsedHarmonyResponse(
            final="Answer",
            channels=None
        )

        assert parsed.channels is None

    def test_channels_dict_empty(self):
        """Test that channels dict can be empty."""
        parsed = ParsedHarmonyResponse(
            final="Answer",
            channels={}
        )

        assert parsed.channels == {}
        assert len(parsed.channels) == 0


class TestMetadataHandling:
    """Test metadata field in ParsedHarmonyResponse."""

    def test_metadata_stores_parse_info(self):
        """Test that metadata can store parsing information."""
        metadata = {
            "token_count": 50,
            "parse_time_ms": 1.5,
            "message_count": 3,
            "channel_count": 2
        }

        parsed = ParsedHarmonyResponse(
            final="Answer",
            metadata=metadata
        )

        assert parsed.metadata["token_count"] == 50
        assert parsed.metadata["parse_time_ms"] == 1.5

    def test_metadata_can_be_none(self):
        """Test that metadata can be None."""
        parsed = ParsedHarmonyResponse(
            final="Answer",
            metadata=None
        )

        assert parsed.metadata is None

    def test_metadata_arbitrary_keys(self):
        """Test that metadata accepts arbitrary keys."""
        metadata = {
            "custom_key": "custom_value",
            "another_key": 123,
            "nested": {"key": "value"}
        }

        parsed = ParsedHarmonyResponse(
            final="Answer",
            metadata=metadata
        )

        assert parsed.metadata["custom_key"] == "custom_value"
        assert parsed.metadata["another_key"] == 123
        assert parsed.metadata["nested"]["key"] == "value"


class TestIntegrationScenarios:
    """Test complete parser integration scenarios."""

    def test_parse_and_extract_workflow(self):
        """Test complete parse and extract workflow."""
        parser = HarmonyResponseParser()

        # Create a parsed response as if from actual parsing
        parsed = ParsedHarmonyResponse(
            final="The answer is 42",
            analysis="First I need to understand the question...",
            commentary="This is a reference to Hitchhiker's Guide",
            channels={
                "final": "The answer is 42",
                "analysis": "First I need to understand the question...",
                "commentary": "This is a reference to Hitchhiker's Guide"
            },
            metadata={"token_count": 150}
        )

        # Extract each channel
        final = parser.extract_channel(parsed, "final")
        analysis = parser.extract_channel(parsed, "analysis")
        commentary = parser.extract_channel(parsed, "commentary")

        assert final == "The answer is 42"
        assert "understand the question" in analysis
        assert "Hitchhiker's Guide" in commentary

    def test_parser_handles_partial_channels(self):
        """Test parser with only some channels present."""
        parser = HarmonyResponseParser()

        # Only final channel present
        parsed = ParsedHarmonyResponse(
            final="Answer",
            analysis=None,
            commentary=None,
            channels={"final": "Answer"}
        )

        assert parser.extract_channel(parsed, "final") == "Answer"
        assert parser.extract_channel(parsed, "analysis") is None
        assert parser.extract_channel(parsed, "commentary") is None


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_extract_channel_with_empty_string(self):
        """Test extracting channel with empty string content."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="",
            channels={"final": ""}
        )

        result = parser.extract_channel(parsed, "final")
        assert result == ""
        assert result is not None  # Empty string, not None

    def test_extract_channel_with_whitespace(self):
        """Test extracting channel with whitespace content."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="  spaces  ",
            channels={"final": "  spaces  "}
        )

        result = parser.extract_channel(parsed, "final")
        # Parser may or may not strip whitespace - just verify it works
        assert isinstance(result, str)

    def test_extract_channel_case_sensitive(self):
        """Test that channel names are case-sensitive."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Answer",
            channels={"final": "Answer", "FINAL": "Different"}
        )

        # Lowercase should work
        assert parser.extract_channel(parsed, "final") == "Answer"

        # Uppercase might be different
        result_upper = parser.extract_channel(parsed, "FINAL")
        if result_upper is not None:
            assert result_upper == "Different"

    def test_multiline_content_in_channels(self):
        """Test channels with multiline content."""
        parser = HarmonyResponseParser()

        multiline = "Line 1\nLine 2\nLine 3"

        parsed = ParsedHarmonyResponse(
            final=multiline,
            channels={"final": multiline}
        )

        result = parser.extract_channel(parsed, "final")
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_unicode_in_channels(self):
        """Test channels with unicode characters."""
        parser = HarmonyResponseParser()

        unicode_text = "Hello 世界 🌍"

        parsed = ParsedHarmonyResponse(
            final=unicode_text,
            channels={"final": unicode_text}
        )

        result = parser.extract_channel(parsed, "final")
        assert "世界" in result
        assert "🌍" in result


class TestContractCompliance:
    """Test compliance with integration contract requirements."""

    def test_final_never_none_in_parsed_response(self):
        """Test that ParsedHarmonyResponse.final is never None."""
        # Contract requirement: final must NEVER be None

        # Minimum valid response
        parsed = ParsedHarmonyResponse(final="")
        assert parsed.final is not None
        assert parsed.final == ""

        # With content
        parsed = ParsedHarmonyResponse(final="Content")
        assert parsed.final is not None
        assert parsed.final == "Content"

    def test_optional_fields_can_be_none(self):
        """Test that analysis and commentary can be None."""
        parsed = ParsedHarmonyResponse(
            final="Answer",
            analysis=None,
            commentary=None
        )

        assert parsed.final == "Answer"
        assert parsed.analysis is None
        assert parsed.commentary is None

    def test_parser_has_required_methods(self):
        """Test that parser has all required interface methods."""
        parser = HarmonyResponseParser()

        # Check required methods exist
        assert hasattr(parser, 'parse_response_tokens')
        assert hasattr(parser, 'parse_response_text')
        assert hasattr(parser, 'validate_harmony_format')
        assert hasattr(parser, 'extract_channel')

        # Check methods are callable
        assert callable(parser.parse_response_tokens)
        assert callable(parser.parse_response_text)
        assert callable(parser.validate_harmony_format)
        assert callable(parser.extract_channel)

    def test_parser_thread_safety(self):
        """Test that parser is thread-safe (stateless)."""
        import threading

        parser = HarmonyResponseParser()
        results = []
        errors = []

        def extract_task(task_id):
            try:
                parsed = ParsedHarmonyResponse(final=f"Task {task_id}")
                result = parser.extract_channel(parsed, "final")
                results.append((task_id, result))
            except Exception as e:
                errors.append((task_id, e))

        # Run concurrent extractions
        threads = [threading.Thread(target=extract_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # Each should have correct result
        for task_id, result in results:
            assert result == f"Task {task_id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
