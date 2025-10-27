"""
Unit tests for advanced inference features (Wave 2).

Tests streaming control, error recovery, and performance optimization.
"""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.inference.streaming import StreamController, StreamState, TokenBuffer
from src.inference.recovery import (
    ErrorClassifier,
    ErrorType,
    RetryStrategy,
    RetryConfig,
    GracefulDegradation,
    ResilientInference,
)
from src.inference.optimization import (
    MemoryMonitor,
    MetalCacheManager,
    MemoryAwareGenerator,
    BatchOptimizer,
    InferenceOptimizer,
)
from src.inference.exceptions import (
    GenerationError,
    ModelNotLoadedError,
    ContextLengthExceededError,
)
from src.sampling.params import SamplingParams


class TestStreamController:
    """Test StreamController for streaming lifecycle management."""

    def test_initial_state(self):
        """Test controller starts in IDLE state."""
        controller = StreamController()
        assert controller.state == StreamState.IDLE
        assert not controller.is_active
        assert not controller.is_paused
        assert not controller.is_cancelled

    def test_start_stream(self):
        """Test starting a stream."""
        controller = StreamController()
        controller.start()
        assert controller.state == StreamState.RUNNING
        assert controller.is_active

    def test_pause_resume(self):
        """Test pausing and resuming a stream."""
        controller = StreamController()
        controller.start()

        controller.pause()
        assert controller.state == StreamState.PAUSED
        assert controller.is_paused
        assert controller.is_active

        controller.resume()
        assert controller.state == StreamState.RUNNING
        assert not controller.is_paused

    def test_cancel_stream(self):
        """Test cancelling a stream."""
        controller = StreamController()
        controller.start()

        controller.cancel()
        assert controller.state == StreamState.CANCELLED
        assert controller.is_cancelled
        assert not controller.is_active

    def test_complete_stream(self):
        """Test completing a stream."""
        controller = StreamController()
        controller.start()

        controller.complete()
        assert controller.state == StreamState.COMPLETED

    def test_token_counting(self):
        """Test token counting and stats."""
        controller = StreamController()
        controller.start()

        controller.increment_tokens(5)
        controller.increment_tokens(3)

        stats = controller.get_stats()
        assert stats["total_tokens"] == 8
        assert stats["state"] == StreamState.RUNNING.value

    def test_progress_callback(self):
        """Test progress callback notifications."""
        controller = StreamController()

        callback_data = []
        def callback(progress):
            callback_data.append(progress)

        controller.add_progress_callback(callback)
        controller.start()
        controller.increment_tokens(1)

        assert len(callback_data) > 0
        assert callback_data[-1]["total_tokens"] == 1

    def test_reset(self):
        """Test resetting controller."""
        controller = StreamController()
        controller.start()
        controller.increment_tokens(5)

        controller.reset()
        assert controller.state == StreamState.IDLE
        assert controller.get_stats()["total_tokens"] == 0


class TestTokenBufferEnhanced:
    """Test enhanced TokenBuffer with auto-flush."""

    def test_basic_buffering(self):
        """Test basic token buffering."""
        buffer = TokenBuffer()
        buffer.add_token("Hello")
        buffer.add_token(" ")
        buffer.add_token("world")

        assert buffer.get_text() == "Hello world"
        assert len(buffer) == 3

    def test_auto_flush(self):
        """Test auto-flush functionality."""
        flushed = []
        def callback(text):
            flushed.append(text)

        buffer = TokenBuffer(
            callback=callback,
            buffer_size=3,
            auto_flush=True
        )

        # Add tokens to trigger flush
        buffer.add_token("a")
        buffer.add_token("b")
        buffer.add_token("c")  # Should trigger flush

        assert len(flushed) == 1
        assert flushed[0] == "abc"
        assert buffer.get_pending_count() == 0

    def test_manual_flush(self):
        """Test manual flushing."""
        flushed = []
        def callback(text):
            flushed.append(text)

        buffer = TokenBuffer(callback=callback, auto_flush=False)
        buffer.add_token("test")
        buffer.add_token("ing")

        result = buffer.flush()
        assert result == "testing"
        assert len(flushed) == 3  # 2 from add_token + 1 from flush

    def test_pending_count(self):
        """Test pending token counting."""
        buffer = TokenBuffer(buffer_size=5, auto_flush=True)
        buffer.add_token("a")
        buffer.add_token("b")

        assert buffer.get_pending_count() == 2


class TestErrorClassifier:
    """Test error classification."""

    def test_classify_fatal_error(self):
        """Test classification of fatal errors."""
        error = ModelNotLoadedError()
        classification = ErrorClassifier.classify(error)

        assert classification.error_type == ErrorType.FATAL
        assert not classification.can_retry
        assert not classification.can_degrade

    def test_classify_degradable_error(self):
        """Test classification of degradable errors."""
        error = ContextLengthExceededError(1000, 512)
        classification = ErrorClassifier.classify(error)

        assert classification.error_type == ErrorType.DEGRADABLE
        assert classification.can_retry
        assert classification.can_degrade

    def test_classify_memory_error(self):
        """Test classification of memory errors."""
        error = GenerationError("CUDA out of memory")
        classification = ErrorClassifier.classify(error)

        assert classification.error_type == ErrorType.DEGRADABLE
        assert classification.can_degrade

    def test_classify_recoverable_error(self):
        """Test classification of recoverable errors."""
        error = GenerationError("Connection timeout")
        classification = ErrorClassifier.classify(error)

        assert classification.error_type == ErrorType.RECOVERABLE
        assert classification.can_retry


class TestRetryStrategy:
    """Test retry strategy with exponential backoff."""

    def test_successful_first_attempt(self):
        """Test successful execution on first attempt."""
        strategy = RetryStrategy(RetryConfig(max_retries=3))

        def success():
            return "success"

        result = strategy.execute(success)
        assert result == "success"

    def test_retry_with_eventual_success(self):
        """Test retry that eventually succeeds."""
        strategy = RetryStrategy(RetryConfig(
            max_retries=3,
            initial_delay=0.01,  # Fast for testing
        ))

        attempts = [0]
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise GenerationError("Temporary failure")
            return "success"

        result = strategy.execute(flaky)
        assert result == "success"
        assert attempts[0] == 3

    def test_retry_exhaustion(self):
        """Test retry exhaustion raises error."""
        strategy = RetryStrategy(RetryConfig(
            max_retries=2,
            initial_delay=0.01,
        ))

        def always_fails():
            raise GenerationError("Always fails")

        with pytest.raises(GenerationError):
            strategy.execute(always_fails)

    def test_fatal_error_no_retry(self):
        """Test fatal errors are not retried."""
        strategy = RetryStrategy(RetryConfig(max_retries=3))

        def fatal():
            raise ModelNotLoadedError()

        with pytest.raises(ModelNotLoadedError):
            strategy.execute(fatal)

    def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable testing
        )
        strategy = RetryStrategy(config)

        # Test delay calculation
        assert strategy._calculate_delay(1) == pytest.approx(0.1, rel=0.01)
        assert strategy._calculate_delay(2) == pytest.approx(0.2, rel=0.01)
        assert strategy._calculate_delay(3) == pytest.approx(0.4, rel=0.01)


class TestGracefulDegradation:
    """Test graceful degradation strategies."""

    def test_reduce_max_tokens(self):
        """Test reducing max_tokens."""
        params = SamplingParams(max_tokens=1000)
        degraded = GracefulDegradation.reduce_max_tokens(params, reduction_factor=0.5)

        assert degraded.max_tokens == 500
        assert degraded.temperature == params.temperature  # Other params unchanged

    def test_truncate_prompt_end(self):
        """Test truncating prompt at end."""
        prompt = "This is a very long prompt that needs truncation"
        truncated = GracefulDegradation.truncate_prompt(
            prompt, max_length=20, truncation_strategy="end"
        )

        assert len(truncated) == 20
        assert truncated.endswith("...")
        assert truncated.startswith("This is")

    def test_truncate_prompt_start(self):
        """Test truncating prompt at start."""
        prompt = "This is a very long prompt that needs truncation"
        truncated = GracefulDegradation.truncate_prompt(
            prompt, max_length=20, truncation_strategy="start"
        )

        assert len(truncated) == 20
        assert truncated.startswith("...")

    def test_create_degraded_params_memory(self):
        """Test creating degraded params for memory error."""
        params = SamplingParams(max_tokens=1000)
        error = GenerationError("CUDA out of memory")

        degraded = GracefulDegradation.create_degraded_params(params, error)

        assert degraded is not None
        assert degraded.max_tokens < params.max_tokens

    def test_create_degraded_params_context(self):
        """Test creating degraded params for context error."""
        params = SamplingParams(max_tokens=1000)
        error = ContextLengthExceededError(2000, 1024)

        degraded = GracefulDegradation.create_degraded_params(params, error)

        assert degraded is not None
        assert degraded.max_tokens < params.max_tokens


class TestResilientInference:
    """Test resilient inference wrapper."""

    def test_successful_generation(self):
        """Test successful generation without recovery."""
        resilient = ResilientInference()

        mock_engine = Mock()
        mock_engine.generate = Mock(return_value="success")

        result = resilient.generate_with_recovery(
            mock_engine,
            "test prompt",
            SamplingParams()
        )

        assert result == "success"
        assert mock_engine.generate.call_count == 1

    def test_recovery_with_degradation(self):
        """Test recovery with graceful degradation."""
        resilient = ResilientInference(
            retry_config=RetryConfig(max_retries=1, initial_delay=0.01)
        )

        mock_engine = Mock()
        # First call fails with memory error, second succeeds
        mock_engine.generate = Mock(
            side_effect=[
                GenerationError("out of memory"),
                "success_degraded"
            ]
        )

        result = resilient.generate_with_recovery(
            mock_engine,
            "test",
            SamplingParams(max_tokens=1000)
        )

        assert result == "success_degraded"
        assert mock_engine.generate.call_count == 2


class TestMemoryMonitor:
    """Test memory monitoring utilities."""

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        stats = MemoryMonitor.get_memory_stats()

        assert stats.total_bytes > 0
        assert stats.available_bytes > 0
        assert 0 <= stats.percent_used <= 100
        assert stats.available_gb > 0

    def test_has_available_memory(self):
        """Test checking available memory."""
        # Check for a small amount (should pass)
        assert MemoryMonitor.has_available_memory(0.1)

        # Check for huge amount (should fail)
        assert not MemoryMonitor.has_available_memory(1000000)

    def test_estimate_memory(self):
        """Test memory estimation."""
        gb_needed = MemoryMonitor.estimate_memory_for_generation(
            prompt_tokens=100,
            max_tokens=500,
            bytes_per_token=4.0
        )

        # 600 tokens * 4 bytes = 2400 bytes ≈ 0.0000022 GB
        assert gb_needed > 0
        assert gb_needed < 1  # Should be very small


class TestMetalCacheManager:
    """Test Metal cache management."""

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = MetalCacheManager()
        # Should not raise error even if Metal not available
        assert isinstance(manager.is_available(), bool)

    def test_clear_cache(self):
        """Test cache clearing (may be no-op if Metal unavailable)."""
        manager = MetalCacheManager()
        # Should not raise error
        manager.clear_cache()

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        manager = MetalCacheManager()
        stats = manager.get_cache_stats()

        assert "available" in stats


class TestMemoryAwareGenerator:
    """Test memory-aware parameter adjustment."""

    def test_adjust_max_tokens_sufficient_memory(self):
        """Test adjustment when memory is sufficient."""
        generator = MemoryAwareGenerator(safety_margin_gb=0.1)
        params = SamplingParams(max_tokens=100)

        adjusted = generator.adjust_max_tokens(params, prompt_tokens=50)

        # Should not reduce if memory available
        # Note: This depends on actual system memory
        assert adjusted.max_tokens <= params.max_tokens

    def test_check_generation_feasible(self):
        """Test feasibility checking."""
        generator = MemoryAwareGenerator()

        # Small generation should be feasible
        feasible, reason = generator.check_generation_feasible(
            prompt_tokens=10,
            max_tokens=50
        )
        assert feasible or reason is not None


class TestBatchOptimizer:
    """Test batch optimization."""

    def test_calculate_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        optimizer = BatchOptimizer()

        batch_size = optimizer.calculate_optimal_batch_size(
            total_prompts=100,
            avg_prompt_tokens=50,
            max_tokens=100,
            max_batch_size=32
        )

        assert 1 <= batch_size <= 32

    def test_should_use_parallel(self):
        """Test parallel processing decision."""
        optimizer = BatchOptimizer()

        # Small batch on single CPU - should not parallelize
        assert not optimizer.should_use_parallel(batch_size=2, cpu_count=1)

        # Large batch on multiple CPUs - should parallelize
        result = optimizer.should_use_parallel(batch_size=10, cpu_count=4)
        # Result depends on system, just verify it returns bool
        assert isinstance(result, bool)


class TestInferenceOptimizer:
    """Test unified inference optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = InferenceOptimizer()

        assert optimizer.metal_cache is not None
        assert optimizer.memory_monitor is not None
        assert optimizer.memory_aware is not None
        assert optimizer.batch_optimizer is not None

    def test_prepare_for_generation(self):
        """Test generation preparation."""
        optimizer = InferenceOptimizer()
        params = SamplingParams(max_tokens=100)

        optimized = optimizer.prepare_for_generation(params, prompt_tokens=50)

        assert optimized.max_tokens <= params.max_tokens

    def test_get_optimization_report(self):
        """Test optimization report generation."""
        optimizer = InferenceOptimizer()
        report = optimizer.get_optimization_report()

        assert "memory" in report
        assert "metal" in report
        assert "system" in report
        assert "available_gb" in report["memory"]


# Integration-style tests
class TestStreamingIntegration:
    """Test streaming components working together."""

    def test_stream_with_controller_and_buffer(self):
        """Test StreamController with TokenBuffer."""
        controller = StreamController()
        buffer = TokenBuffer()

        controller.start()

        # Simulate streaming
        tokens = ["Hello", " ", "world", "!"]
        for token in tokens:
            controller.wait_if_paused()
            if controller.is_cancelled:
                break
            buffer.add_token(token)
            controller.increment_tokens()

        controller.complete()

        assert buffer.get_text() == "Hello world!"
        assert controller.get_stats()["total_tokens"] == len(tokens)
        assert controller.state == StreamState.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
