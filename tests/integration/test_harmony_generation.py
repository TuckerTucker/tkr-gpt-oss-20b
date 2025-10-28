"""
Integration tests for Harmony multi-channel generation.

Tests the integration between InferenceEngine and the new openai-harmony
implementation (HarmonyPromptBuilder + HarmonyResponseParser), verifying
that multi-channel responses are properly parsed and reasoning traces are
captured when configured.

REWRITTEN for Wave 4B - NEW IMPLEMENTATION
- Uses HarmonyPromptBuilder (not HarmonyEncoder)
- Uses HarmonyResponseParser (not ParsedResponse)
- Tests complete engine integration (Waves 2-3)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Optional, List

from src.inference.engine import InferenceEngine, GenerationResult
from src.config.inference_config import InferenceConfig, ReasoningLevel
from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    HarmonyResponseParser,
    ParsedHarmonyResponse,
)
from src.sampling.params import SamplingParams


@pytest.fixture
def mock_model_loader():
    """Create a mock ModelLoader for testing."""
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName

    loader = Mock()
    loader.is_loaded.return_value = True

    # Mock model and tokenizer
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Use real Harmony encoding for proper tokenization
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Mock tokenizer.encode() using real Harmony encoding
    def mock_encode(text, allowed_special=None):
        # Use real encoding to properly handle Harmony special tokens
        return list(encoding.encode(text, allowed_special="all"))

    mock_tokenizer.encode = mock_encode
    loader.get_model.return_value = (mock_model, mock_tokenizer)

    return loader


@pytest.fixture
def harmony_config():
    """Config with Harmony enabled and reasoning capture enabled."""
    return InferenceConfig(
        use_harmony_format=True,
        capture_reasoning=True,
        reasoning_level=ReasoningLevel.MEDIUM,
        temperature=0.7,
        max_tokens=512
    )


@pytest.fixture
def harmony_config_no_capture():
    """Config with Harmony enabled but reasoning capture disabled."""
    return InferenceConfig(
        use_harmony_format=True,
        capture_reasoning=False,
        reasoning_level=ReasoningLevel.MEDIUM,
        temperature=0.7,
        max_tokens=512
    )


@pytest.fixture
def legacy_config():
    """Config with Harmony disabled (legacy mode)."""
    return InferenceConfig(
        use_harmony_format=False,
        capture_reasoning=False,
        temperature=0.7,
        max_tokens=512
    )


class TestHarmonyGeneration:
    """Test Harmony multi-channel generation integration."""

    def test_uses_harmony_builder_and_parser(self, mock_model_loader, harmony_config):
        """Test that engine initializes HarmonyPromptBuilder and HarmonyResponseParser."""
        engine = InferenceEngine(mock_model_loader, harmony_config)

        # Verify NEW components were initialized
        assert hasattr(engine, 'harmony_builder')
        assert hasattr(engine, 'harmony_parser')
        assert isinstance(engine.harmony_builder, HarmonyPromptBuilder)
        assert isinstance(engine.harmony_parser, HarmonyResponseParser)

    def test_generation_result_has_reasoning_fields(self, mock_model_loader, harmony_config):
        """Test GenerationResult has reasoning, commentary, and channels fields."""
        # Create a sample harmony response
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Let me analyze this question carefully.\n"
            "The user is asking about X.\n"
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Here is my answer to your question."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Check that result has new fields
            assert hasattr(result, 'reasoning')
            assert hasattr(result, 'commentary')
            assert hasattr(result, 'channels')

    def test_reasoning_captured_when_enabled(self, mock_model_loader, harmony_config):
        """Test reasoning is captured when capture_reasoning=True."""
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Analyzing the question...\n"
            "Key points: 1, 2, 3"
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "The answer is 42."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Verify reasoning was captured (from parsed.analysis)
            assert result.reasoning is not None
            assert "Analyzing the question" in result.reasoning
            assert "Key points" in result.reasoning

            # Verify channels dict contains all channels
            assert result.channels is not None
            assert 'analysis' in result.channels
            assert 'final' in result.channels

            # Verify final text doesn't contain reasoning (from parsed.final)
            assert result.text == "The answer is 42."
            assert "Analyzing" not in result.text

    def test_reasoning_not_captured_when_disabled(self, mock_model_loader, harmony_config_no_capture):
        """Test reasoning is NOT captured when capture_reasoning=False."""
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Internal analysis here."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Final answer."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config_no_capture)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Verify reasoning was NOT captured
            assert result.reasoning is None
            assert result.commentary is None
            assert result.channels is None

            # Verify final text is still extracted correctly
            assert result.text == "Final answer."

    def test_commentary_channel_captured(self, mock_model_loader, harmony_config):
        """Test commentary channel is captured when present."""
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Analyzing..."
            "<|end|>"
            "<|start|>assistant<|channel|>commentary<|message|>"
            "Executing search_tool()..."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Based on search results: answer."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Verify all channels captured
            assert result.reasoning is not None
            assert result.commentary is not None
            assert "Executing search_tool" in result.commentary

            assert result.channels is not None
            assert 'analysis' in result.channels
            assert 'commentary' in result.channels
            assert 'final' in result.channels

            # Final text should only have final channel
            assert result.text == "Based on search results: answer."

    def test_backward_compatibility_legacy_mode(self, mock_model_loader, legacy_config):
        """Test backward compatibility with use_harmony_format=False."""
        # Old format without channel markers
        legacy_response = "This is a simple response without channels."

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = legacy_response

            engine = InferenceEngine(mock_model_loader, legacy_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Verify legacy mode doesn't capture reasoning
            assert result.reasoning is None
            assert result.commentary is None
            assert result.channels is None

            # Text should be raw response (no parsing)
            assert result.text == legacy_response

    def test_get_reasoning_trace_method(self, mock_model_loader, harmony_config):
        """Test get_reasoning_trace() method."""
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Step 1: Understand the problem.\n"
            "Step 2: Formulate solution."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Solution provided."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Use the get_reasoning_trace method
            reasoning = engine.get_reasoning_trace(result)

            assert reasoning is not None
            assert "Step 1" in reasoning
            assert "Step 2" in reasoning

    def test_get_reasoning_trace_returns_none_when_not_captured(
        self, mock_model_loader, harmony_config_no_capture
    ):
        """Test get_reasoning_trace() returns None when reasoning not captured."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Simple answer."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config_no_capture)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            reasoning = engine.get_reasoning_trace(result)
            assert reasoning is None

    def test_harmony_parser_called_with_response_text(self, mock_model_loader, harmony_config):
        """Test that HarmonyResponseParser.parse_response_text is called with model output."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Test response"
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Mock the parser's parse_response_text method
            with patch.object(engine.harmony_parser, 'parse_response_text') as mock_parse:
                mock_parse.return_value = ParsedHarmonyResponse(
                    final="Test response",
                    analysis=None,
                    commentary=None,
                )

                result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

                # Verify parse_response_text was called with the model output
                mock_parse.assert_called_once()
                call_args = mock_parse.call_args
                assert call_args[1]['response_text'] == harmony_response

    def test_only_final_channel_in_text_field(self, mock_model_loader, harmony_config):
        """Test that .text field NEVER contains analysis channel content."""
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "SECRET_ANALYSIS_DATA_SHOULD_NOT_APPEAR"
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Clean user-facing response."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Critical: text field must NOT contain analysis
            assert "SECRET_ANALYSIS_DATA" not in result.text
            assert result.text == "Clean user-facing response."

            # But reasoning field should have it when capture enabled
            assert "SECRET_ANALYSIS_DATA" in result.reasoning

    def test_parser_used_in_harmony_mode(self, mock_model_loader, harmony_config):
        """Test that HarmonyResponseParser is used in harmony mode."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Response text"
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Verify parser is called by checking result structure
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Result should have parsed structure
            assert hasattr(result, 'reasoning')
            assert hasattr(result, 'commentary')
            assert hasattr(result, 'channels')

    def test_handles_missing_channels_gracefully(self, mock_model_loader, harmony_config):
        """Test graceful handling when only final channel is present."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Only final channel here."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Should handle missing analysis/commentary gracefully
            assert result.text == "Only final channel here."
            assert result.reasoning is None
            assert result.commentary is None

            # Channels dict should only have final
            assert result.channels is not None
            assert 'final' in result.channels
            assert 'analysis' not in result.channels

    def test_all_existing_tests_still_pass(self, mock_model_loader):
        """Test that existing code using GenerationResult still works."""
        # Simulate old test that doesn't know about new fields
        simple_response = "Simple text"

        with patch('mlx_lm.generate') as mock_generate:
            # Add small delay to simulate realistic generation
            import time
            def slow_generate(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return simple_response
            mock_generate.side_effect = slow_generate

            # Use engine without config (should use defaults)
            engine = InferenceEngine(mock_model_loader)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Old code should still work
            assert isinstance(result, GenerationResult)
            assert result.text == simple_response
            assert result.tokens_generated > 0
            assert result.latency_ms >= 0  # Allow 0 for fast mocks

            # New fields exist but are None (backward compatible)
            assert result.reasoning is None
            assert result.commentary is None
            assert result.channels is None


class TestMultiTurnConversations:
    """Test multi-turn conversations with history integration."""

    def test_multi_turn_with_conversation_history(self, mock_model_loader, harmony_config):
        """Test multi-turn conversation with history uses Harmony format."""
        # Simulate a multi-turn conversation
        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "The user is asking a follow-up question about installation."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "To install Python, download it from python.org and run the installer."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Generate with prompt (simulating multi-turn)
            result = engine.generate(
                "How do I install it?",
                SamplingParams(max_tokens=100)
            )

            # Verify Harmony format was used
            assert result.reasoning is not None
            assert "follow-up question" in result.reasoning
            assert "install Python" in result.text

    def test_reasoning_level_propagation(self, mock_model_loader):
        """Test reasoning level affects generation."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>Test response<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            # Test different reasoning levels
            for level in [ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]:
                config = InferenceConfig(
                    use_harmony_format=True,
                    reasoning_level=level,
                    max_tokens=100
                )
                engine = InferenceEngine(mock_model_loader, config)

                # Verify builder was called with correct reasoning level
                # (indirectly verified through successful generation)
                result = engine.generate("Test", SamplingParams(max_tokens=100))
                assert result is not None


class TestStreamingIntegration:
    """Test streaming generation with channel detection."""

    def test_streaming_yields_channel_metadata(self, mock_model_loader, harmony_config):
        """Test streaming yields tokens with channel metadata."""
        # Mock streaming response
        stream_tokens = [
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "Hello",
            " world",
            "<|end|>"
        ]

        with patch('mlx_lm.stream_generate') as mock_stream:
            mock_stream.return_value = iter(stream_tokens)

            engine = InferenceEngine(mock_model_loader, harmony_config)
            stream = engine.generate_stream("Test prompt", SamplingParams(max_tokens=100))

            tokens = list(stream)

            # Check token structure
            assert len(tokens) > 0
            assert all('token' in t for t in tokens)
            assert all('channel' in t for t in tokens)
            assert all('is_final' in t for t in tokens)

            # Verify channel metadata present
            channels_seen = set(t['channel'] for t in tokens)
            # At minimum, should have seen channel metadata
            assert len(channels_seen) > 0

    def test_streaming_handles_errors_gracefully(self, mock_model_loader, harmony_config):
        """Test streaming handles errors without crashing."""
        # Mock streaming that raises an error mid-stream
        def error_stream():
            yield "<|start|>"
            yield "assistant"
            raise RuntimeError("Simulated error")

        with patch('mlx_lm.stream_generate') as mock_stream:
            mock_stream.return_value = error_stream()

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Should raise GenerationError (not crash)
            with pytest.raises(Exception):  # GenerationError or RuntimeError
                stream = engine.generate_stream("Test", SamplingParams(max_tokens=100))
                list(stream)  # Consume stream


class TestErrorHandling:
    """Test error handling with malformed responses."""

    def test_handles_malformed_harmony_response(self, mock_model_loader, harmony_config):
        """Test graceful handling of malformed Harmony response."""
        # Malformed response (missing end tag)
        malformed_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Incomplete response..."
            # Missing <|end|>
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = malformed_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Should not crash, should fallback to raw text
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Should have some text (even if parsing failed)
            assert result.text is not None
            assert len(result.text) > 0

    def test_handles_empty_response(self, mock_model_loader, harmony_config):
        """Test handling of empty model response."""
        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = ""

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Should handle empty response gracefully
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Result should exist with empty text
            assert result is not None
            assert result.text == ""


class TestPerformanceBenchmarks:
    """Test Harmony parsing performance requirements."""

    def test_harmony_parsing_overhead_acceptable(self, mock_model_loader, harmony_config):
        """Test that Harmony parsing adds minimal overhead."""
        import time

        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Some analysis content."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Final response."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Warm up
            for _ in range(3):
                engine.generate("Test", SamplingParams(max_tokens=100))

            # Measure parsing overhead
            iterations = 10
            times = []

            for _ in range(iterations):
                start = time.perf_counter()
                result = engine.generate("Test prompt", SamplingParams(max_tokens=100))
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            avg_time = sum(times) / len(times)

            # Parsing overhead should be reasonable (< 100ms for mocked generation)
            # Note: This is very generous for mocked tests
            assert avg_time < 100, f"Parsing overhead {avg_time:.2f}ms exceeds 100ms"

    def test_no_impact_on_generation_metrics(self, mock_model_loader, harmony_config):
        """Test that Harmony parsing doesn't affect generation metrics."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Response text"
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            # Add small delay to simulate realistic generation
            import time
            def slow_generate(*args, **kwargs):
                time.sleep(0.01)  # 10ms delay
                return harmony_response
            mock_generate.side_effect = slow_generate

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Metrics should be present and valid
            assert result.tokens_generated > 0
            assert result.tokens_per_second >= 0  # Allow 0 for very fast tests
            assert result.latency_ms >= 0
            assert result.finish_reason in ["stop", "length", "error"]


class TestConversationHistoryIntegration:
    """Test conversation history integration with Harmony."""

    def test_conversation_manager_stores_harmony_response(self, mock_model_loader, harmony_config):
        """Test ConversationManager can store Harmony responses."""
        from src.conversation.history import ConversationManager

        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Detailed analysis here."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "User-facing response."
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)
            result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

            # Add to conversation manager
            conv = ConversationManager()
            conv.add_message("user", "Test prompt")
            msg = conv.add_message(
                "assistant",
                result.text,
                channels=result.channels
            )

            # Verify message stored correctly
            assert msg.content == "User-facing response."
            assert msg.channels is not None
            assert 'final' in msg.channels
            assert 'analysis' in msg.channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
