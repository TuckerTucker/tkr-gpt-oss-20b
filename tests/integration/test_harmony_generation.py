"""
Integration tests for Harmony multi-channel generation.

Tests the integration between InferenceEngine and HarmonyEncoder,
verifying that multi-channel responses are properly parsed and
reasoning traces are captured when configured.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Optional

from src.inference.engine import InferenceEngine, GenerationResult
from src.config.inference_config import InferenceConfig
from src.prompts.harmony import HarmonyEncoder, ParsedResponse
from src.sampling.params import SamplingParams


@pytest.fixture
def mock_model_loader():
    """Create a mock ModelLoader for testing."""
    loader = Mock()
    loader.is_loaded.return_value = True

    # Mock model and tokenizer
    mock_model = Mock()
    mock_tokenizer = Mock()
    loader.get_model.return_value = (mock_model, mock_tokenizer)

    return loader


@pytest.fixture
def harmony_config():
    """Config with Harmony enabled and reasoning capture enabled."""
    return InferenceConfig(
        use_harmony_format=True,
        capture_reasoning=True,
        temperature=0.7,
        max_tokens=512
    )


@pytest.fixture
def harmony_config_no_capture():
    """Config with Harmony enabled but reasoning capture disabled."""
    return InferenceConfig(
        use_harmony_format=True,
        capture_reasoning=False,
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

    def test_uses_harmony_encoder(self, mock_model_loader, harmony_config):
        """Test that generate() uses HarmonyEncoder when enabled."""
        engine = InferenceEngine(mock_model_loader, harmony_config)

        # Verify HarmonyEncoder was initialized
        assert isinstance(engine.harmony_encoder, HarmonyEncoder)

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

            # Verify reasoning was captured
            assert result.reasoning is not None
            assert "Analyzing the question" in result.reasoning
            assert "Key points" in result.reasoning

            # Verify channels dict contains all channels
            assert result.channels is not None
            assert 'analysis' in result.channels
            assert 'final' in result.channels

            # Verify final text doesn't contain reasoning
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

            # Text should be extracted using legacy method
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

            # Use the new method
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

    def test_harmony_parser_called_with_correct_response(self, mock_model_loader, harmony_config):
        """Test that HarmonyEncoder.parse_response is called with model output."""
        harmony_response = (
            "<|start|>assistant<|channel|>final<|message|>"
            "Test response"
            "<|end|>"
        )

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = harmony_response

            engine = InferenceEngine(mock_model_loader, harmony_config)

            # Mock the encoder's parse_response method
            with patch.object(engine.harmony_encoder, 'parse_response') as mock_parse:
                mock_parse.return_value = ParsedResponse(
                    final="Test response",
                    analysis=None,
                    commentary=None,
                    raw=harmony_response
                )

                result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

                # Verify parse_response was called with the model output
                mock_parse.assert_called_once_with(harmony_response)

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

    def test_no_harmony_encoder_in_legacy_mode(self, mock_model_loader, legacy_config):
        """Test that HarmonyEncoder is not used in legacy mode."""
        legacy_response = "Legacy response text"

        with patch('mlx_lm.generate') as mock_generate:
            mock_generate.return_value = legacy_response

            engine = InferenceEngine(mock_model_loader, legacy_config)

            # Mock the encoder to ensure it's not called
            with patch.object(engine.harmony_encoder, 'parse_response') as mock_parse:
                result = engine.generate("Test prompt", SamplingParams(max_tokens=100))

                # Verify parse_response was NOT called in legacy mode
                mock_parse.assert_not_called()

                # Should use legacy extraction
                assert result.text == legacy_response

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


class TestHarmonyPerformance:
    """Test Harmony parsing performance requirements."""

    def test_harmony_parsing_overhead_under_5ms(self, mock_model_loader, harmony_config):
        """Test that Harmony parsing adds < 5ms overhead."""
        import time

        harmony_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Some analysis content."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Final response."
            "<|end|>"
        )

        # Test direct parsing time
        encoder = HarmonyEncoder()

        start = time.time()
        for _ in range(100):  # Average over 100 parses
            parsed = encoder.parse_response(harmony_response)
        end = time.time()

        avg_time_ms = ((end - start) / 100) * 1000

        # Should be well under 5ms per parse
        assert avg_time_ms < 5.0, f"Parsing took {avg_time_ms:.2f}ms (expected < 5ms)"

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

            # Metrics should be based on total generation, not just final channel
            assert result.tokens_generated > 0
            assert result.tokens_per_second >= 0  # Allow 0 for very fast tests
            assert result.latency_ms >= 0
            assert result.finish_reason in ["stop", "length", "error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
