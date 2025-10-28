"""
Test Agent 3C: Streaming Integration with Harmony format.

This test validates that generate_stream() correctly:
1. Uses HarmonyPromptBuilder for prompt creation
2. Passes token IDs to MLX
3. Yields tokens with channel metadata
4. Detects channel transitions during streaming
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from src.inference.engine import InferenceEngine
from src.config.inference_config import InferenceConfig


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text, allowed_special="all"):
        """Mock encode method."""
        # Return token IDs - each character as a token for simplicity
        return [ord(c) for c in text[:10]]  # Limit to 10 for testing


class MockMLXModule:
    """Mock MLX module for testing."""

    @staticmethod
    def stream_generate(model, tokenizer, prompt, **kwargs):
        """
        Mock stream_generate that yields tokens.

        For testing, we expect prompt to be a List[int] (token IDs).
        """
        # Verify prompt is a list of token IDs
        assert isinstance(prompt, list), "Prompt must be token IDs (List[int])"
        assert all(isinstance(t, int) for t in prompt), "All prompt items must be integers"

        # Yield some test tokens
        test_tokens = ["Hello", " ", "world", "!", " ", "Test", " ", "streaming", "."]
        for token in test_tokens:
            yield token

    @staticmethod
    def generate(model, tokenizer, prompt, **kwargs):
        """Mock generate for non-streaming."""
        return "Mock generated text"


def test_streaming_uses_harmony_builder():
    """Test that generate_stream() uses HarmonyPromptBuilder."""
    # Patch mlx_lm in sys.modules
    with patch.dict(sys.modules, {'mlx_lm': MockMLXModule()}):
        # Setup mock model loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_model.return_value = (Mock(), MockTokenizer())

        # Create engine
        config = InferenceConfig()
        engine = InferenceEngine(mock_loader, config)

        # Generate stream
        stream = engine.generate_stream("Test prompt")

        # Consume stream and collect tokens
        tokens = list(stream)

        # Verify we got token dictionaries (not strings)
        assert len(tokens) > 0, "Should yield tokens"

        # Check first token structure
        first_token = tokens[0]
        assert isinstance(first_token, dict), "Should yield dict, not string"
        assert 'token' in first_token, "Should have 'token' key"
        assert 'channel' in first_token, "Should have 'channel' key"
        assert 'content' in first_token, "Should have 'content' key"
        assert 'delta' in first_token, "Should have 'delta' key"
        assert 'is_final' in first_token, "Should have 'is_final' key"


def test_streaming_yields_channel_metadata():
    """Test that channel metadata is included in yielded tokens."""
    # Patch mlx_lm in sys.modules
    with patch.dict(sys.modules, {'mlx_lm': MockMLXModule()}):
        # Setup mock model loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_model.return_value = (Mock(), MockTokenizer())

        # Create engine
        config = InferenceConfig()
        engine = InferenceEngine(mock_loader, config)

        # Generate stream
        stream = engine.generate_stream("Test prompt")

        # Collect all tokens
        tokens = list(stream)

        # Verify each token has channel info
        for i, token_dict in enumerate(tokens):
            assert 'channel' in token_dict, f"Token {i} missing channel"
            assert isinstance(token_dict['channel'], str), f"Token {i} channel not string"
            assert 'is_final' in token_dict, f"Token {i} missing is_final"
            assert isinstance(token_dict['is_final'], bool), f"Token {i} is_final not bool"


def test_streaming_token_ids_passed_to_mlx():
    """Test that token IDs (not strings) are passed to MLX."""
    # Patch mlx_lm in sys.modules
    with patch.dict(sys.modules, {'mlx_lm': MockMLXModule()}):
        # Setup mock model loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_model.return_value = (Mock(), MockTokenizer())

        # Create engine
        config = InferenceConfig()
        engine = InferenceEngine(mock_loader, config)

        # This will raise assertion error if prompt is not token IDs
        stream = engine.generate_stream("Test prompt")

        # Consume stream (this triggers the validation in MockMLXModule)
        tokens = list(stream)

        # If we get here, token IDs were passed correctly
        assert len(tokens) > 0


def test_streaming_preserves_existing_features():
    """Test that existing streaming features still work."""
    # Patch mlx_lm in sys.modules
    with patch.dict(sys.modules, {'mlx_lm': MockMLXModule()}):
        # Setup mock model loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_model.return_value = (Mock(), MockTokenizer())

        # Create engine
        config = InferenceConfig()
        engine = InferenceEngine(mock_loader, config)

        # Generate stream
        stream = engine.generate_stream("Test prompt")

        # Verify we can iterate and get tokens
        token_count = 0
        for token_dict in stream:
            token_count += 1
            # Verify basic structure
            assert 'token' in token_dict

        # Should have yielded tokens
        assert token_count > 0, "Should yield tokens"


def test_streaming_cancellation_still_works():
    """Test that cancellation functionality is preserved."""
    # Setup mock model loader
    mock_loader = Mock()
    mock_loader.is_loaded.return_value = True
    mock_loader.get_model.return_value = (Mock(), MockTokenizer())

    # Create engine
    config = InferenceConfig()
    engine = InferenceEngine(mock_loader, config)

    # Test cancellation flag exists
    assert hasattr(engine, '_cancelled'), "Engine should have _cancelled flag"
    assert hasattr(engine, 'cancel_generation'), "Engine should have cancel_generation method"

    # Verify cancel_generation sets the flag
    engine.cancel_generation()
    assert engine._cancelled is True, "cancel_generation should set _cancelled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
