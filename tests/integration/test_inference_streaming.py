"""
Integration tests for streaming inference with advanced features.

Tests the interaction between InferenceEngine, StreamController,
and recovery/optimization systems using mock models.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Iterator

from src.inference import (
    InferenceEngine,
    StreamController,
    StreamState,
    TokenBuffer,
    ResilientInference,
    InferenceOptimizer,
    GenerationError,
)
from src.sampling import SamplingParams


class MockModelLoader:
    """Mock ModelLoader for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model(self):
        return MockModel()

    def get_tokenizer(self):
        return MockTokenizer()


class MockModel:
    """Mock model for testing."""
    pass


class MockTokenizer:
    """Mock tokenizer for testing."""

    def encode(self, text: str):
        # Simple word-based tokenization
        return text.split()

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)


class MockMLXModule:
    """Mock mlx_lm module for testing."""

    @staticmethod
    def generate(model, tokenizer, prompt, **kwargs):
        """Mock synchronous generation."""
        return f"Generated response to: {prompt[:20]}..."

    @staticmethod
    def stream_generate(model, tokenizer, prompt, **kwargs) -> Iterator[str]:
        """Mock streaming generation."""
        response = f"Mock response to prompt"
        for char in response:
            time.sleep(0.001)  # Simulate generation delay
            yield char


@pytest.fixture
def mock_model_loader():
    """Fixture providing mock model loader."""
    return MockModelLoader()


@pytest.fixture
def mock_engine(mock_model_loader):
    """Fixture providing mock inference engine."""
    with patch('src.inference.engine.mlx_lm', MockMLXModule):
        engine = InferenceEngine(mock_model_loader)
        return engine


class TestStreamingWithController:
    """Test streaming generation with StreamController."""

    def test_basic_streaming_with_controller(self, mock_engine):
        """Test basic streaming with controller."""
        controller = StreamController()
        buffer = TokenBuffer()

        controller.start()

        # Mock stream
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            stream = mock_engine.generate_stream("test prompt")

            for token in stream:
                controller.wait_if_paused()
                if controller.is_cancelled:
                    break

                buffer.add_token(token)
                controller.increment_tokens()

        controller.complete()

        assert buffer.get_text() == "Mock response to prompt"
        assert controller.state == StreamState.COMPLETED
        assert controller.get_stats()["total_tokens"] > 0

    def test_streaming_with_pause_resume(self, mock_engine):
        """Test streaming with pause and resume."""
        controller = StreamController()
        buffer = TokenBuffer()
        tokens_collected = []

        controller.start()

        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            stream = mock_engine.generate_stream("test")

            for i, token in enumerate(stream):
                # Pause after 5 tokens
                if i == 5:
                    controller.pause()
                    # Simulate pause duration
                    time.sleep(0.01)
                    controller.resume()

                controller.wait_if_paused()
                buffer.add_token(token)
                controller.increment_tokens()
                tokens_collected.append(token)

        controller.complete()

        # Should have collected all tokens despite pause
        assert len(tokens_collected) > 5
        assert controller.state == StreamState.COMPLETED

    def test_streaming_with_cancellation(self, mock_engine):
        """Test cancelling streaming mid-generation."""
        controller = StreamController()
        buffer = TokenBuffer()

        controller.start()

        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            stream = mock_engine.generate_stream("test")

            for i, token in enumerate(stream):
                if i == 3:
                    controller.cancel()

                if controller.is_cancelled:
                    break

                buffer.add_token(token)
                controller.increment_tokens()

        # Should have stopped early
        assert len(buffer) == 3
        assert controller.is_cancelled

    def test_streaming_with_progress_callback(self, mock_engine):
        """Test streaming with progress callbacks."""
        controller = StreamController()
        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.copy())

        controller.add_progress_callback(on_progress)
        controller.start()

        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            stream = mock_engine.generate_stream("test")

            for token in stream:
                controller.increment_tokens()

        controller.complete()

        # Should have received progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1]["total_tokens"] > 0
        assert "tokens_per_second" in progress_updates[-1]


class TestStreamingWithBuffering:
    """Test streaming with enhanced buffering."""

    def test_auto_flush_buffering(self, mock_engine):
        """Test auto-flush buffering."""
        flushed_chunks = []

        def on_flush(text):
            flushed_chunks.append(text)

        buffer = TokenBuffer(
            callback=on_flush,
            buffer_size=5,
            auto_flush=True
        )

        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            stream = mock_engine.generate_stream("test")

            for token in stream:
                buffer.add_token(token)

        # Final flush for remaining tokens
        buffer.flush()

        # Should have flushed multiple chunks
        assert len(flushed_chunks) > 0
        # Reconstruct full text from chunks
        full_text = "".join(flushed_chunks)
        assert "Mock response" in full_text


class TestResilientStreaming:
    """Test streaming with error recovery."""

    def test_resilient_generation_success(self, mock_model_loader):
        """Test resilient generation with successful first attempt."""
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            engine = InferenceEngine(mock_model_loader)
            resilient = ResilientInference()

            result = resilient.generate_with_recovery(
                engine,
                "test prompt",
                SamplingParams(max_tokens=50)
            )

            assert result is not None

    def test_resilient_generation_with_retry(self, mock_model_loader):
        """Test resilient generation with retry."""
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            engine = InferenceEngine(mock_model_loader)

            # Mock generate to fail once then succeed
            call_count = [0]
            original_generate = engine.generate

            def flaky_generate(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise GenerationError("Temporary failure")
                return original_generate(*args, **kwargs)

            engine.generate = flaky_generate

            resilient = ResilientInference()
            result = resilient.generate_with_recovery(
                engine,
                "test",
                SamplingParams(max_tokens=50)
            )

            # Should have retried and succeeded
            assert call_count[0] == 2
            assert result is not None


class TestOptimizedStreaming:
    """Test streaming with optimization."""

    def test_streaming_with_optimization(self, mock_model_loader):
        """Test streaming with inference optimizer."""
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            engine = InferenceEngine(mock_model_loader)
            optimizer = InferenceOptimizer()

            # Prepare optimized params
            params = SamplingParams(max_tokens=100)
            optimized_params = optimizer.prepare_for_generation(
                params,
                prompt_tokens=50
            )

            # Generate with optimized params
            stream = engine.generate_stream("test", optimized_params)

            tokens = []
            for token in stream:
                tokens.append(token)

            assert len(tokens) > 0

    def test_memory_aware_streaming(self, mock_model_loader):
        """Test memory-aware streaming."""
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            engine = InferenceEngine(mock_model_loader)
            optimizer = InferenceOptimizer()

            # Request large generation
            params = SamplingParams(max_tokens=10000)

            # Should adjust for available memory
            safe_params = optimizer.memory_aware.adjust_max_tokens(
                params,
                prompt_tokens=100
            )

            # Generate with safe params
            result = engine.generate("test", safe_params)

            assert result is not None


class TestFullStreamingPipeline:
    """Test complete streaming pipeline with all features."""

    def test_complete_pipeline(self, mock_model_loader):
        """Test complete streaming pipeline with all enhancements."""
        with patch('src.inference.engine.mlx_lm', MockMLXModule):
            # Setup components
            engine = InferenceEngine(mock_model_loader)
            controller = StreamController()
            optimizer = InferenceOptimizer()

            # Prepare optimized params
            params = SamplingParams(max_tokens=100)
            optimized_params = optimizer.prepare_for_generation(
                params,
                prompt_tokens=50
            )

            # Setup buffering with auto-flush
            collected_chunks = []
            buffer = TokenBuffer(
                callback=lambda text: collected_chunks.append(text),
                buffer_size=5,
                auto_flush=True
            )

            # Setup progress tracking
            progress_updates = []
            controller.add_progress_callback(
                lambda p: progress_updates.append(p["total_tokens"])
            )

            # Start streaming
            controller.start()

            stream = engine.generate_stream("test prompt", optimized_params)

            for token in stream:
                controller.wait_if_paused()
                if controller.is_cancelled:
                    break

                buffer.add_token(token)
                controller.increment_tokens()

            buffer.flush()  # Final flush
            controller.complete()

            # Verify all components worked
            assert controller.state == StreamState.COMPLETED
            assert len(collected_chunks) > 0
            assert len(progress_updates) > 0
            assert buffer.get_text() != ""

            # Get final stats
            stats = controller.get_stats()
            assert stats["total_tokens"] > 0
            assert stats["tokens_per_second"] >= 0

            # Get optimization report
            report = optimizer.get_optimization_report()
            assert "memory" in report
            assert "metal" in report


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_streaming_empty_output(self, mock_model_loader):
        """Test streaming with empty output."""
        with patch('src.inference.engine.mlx_lm.stream_generate', return_value=iter([])):
            engine = InferenceEngine(mock_model_loader)
            controller = StreamController()
            buffer = TokenBuffer()

            controller.start()

            stream = engine.generate_stream("test")
            for token in stream:
                buffer.add_token(token)
                controller.increment_tokens()

            controller.complete()

            assert buffer.get_text() == ""
            assert controller.get_stats()["total_tokens"] == 0

    def test_streaming_with_special_characters(self, mock_model_loader):
        """Test streaming with special characters."""
        special_chars = "Hello\n\t\"'<>\\world"

        def mock_stream(*args, **kwargs):
            for char in special_chars:
                yield char

        with patch('src.inference.engine.mlx_lm.stream_generate', mock_stream):
            engine = InferenceEngine(mock_model_loader)
            buffer = TokenBuffer()

            stream = engine.generate_stream("test")
            for token in stream:
                buffer.add_token(token)

            assert buffer.get_text() == special_chars

    def test_streaming_with_controller_state_transitions(self, mock_model_loader):
        """Test all controller state transitions."""
        controller = StreamController()

        # Test invalid state transitions
        controller.pause()  # Should warn - can't pause from IDLE
        assert controller.state == StreamState.IDLE

        controller.start()
        assert controller.state == StreamState.RUNNING

        controller.pause()
        assert controller.state == StreamState.PAUSED

        controller.resume()
        assert controller.state == StreamState.RUNNING

        controller.cancel()
        assert controller.state == StreamState.CANCELLED

        # Reset and test complete
        controller.reset()
        controller.start()
        controller.complete()
        assert controller.state == StreamState.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
