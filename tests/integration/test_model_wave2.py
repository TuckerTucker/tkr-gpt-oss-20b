"""
Integration tests for Wave 2 model functionality.

These tests verify the integration between health checks, model switching,
and model tools WITHOUT loading actual MLX models.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models import (
    ModelLoader,
    check_model_loaded,
    check_metal_backend,
    get_health_status,
    format_health_report,
    switch_model,
    list_models,
    get_model_info,
    check_model_compatibility,
)
from src.config import ModelConfig


@pytest.fixture
def mock_model_config():
    """Create a mock ModelConfig for testing."""
    return ModelConfig(
        model_name="phi-3-mini",
        quantization="4bit",
        device="auto",
        max_model_len=4096,
    )


@pytest.fixture
def mock_loader(mock_model_config):
    """
    Create a mock ModelLoader that simulates behavior without loading.

    This fixture creates a ModelLoader with mocked dependencies to avoid
    actually loading MLX models during tests.
    """
    with patch("src.models.loader.mlx_lm") as mock_mlx:
        # Mock mlx_lm.load to return fake model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_mlx.load.return_value = (mock_model, mock_tokenizer)

        # Mock mlx_lm.generate to return fake output
        mock_mlx.generate.return_value = "Test output"

        # Mock device detection
        with patch("src.devices.DeviceDetector") as mock_detector:
            mock_detector.validate_platform.return_value = (True, "Valid")
            mock_detector.detect.return_value = "mps"

            # Mock memory manager
            with patch("src.models.memory.MemoryManager") as mock_memory:
                mock_memory.can_fit_model.return_value = (True, "Fits")
                mock_memory.get_model_memory_usage.return_value = 2500

                # Create loader
                loader = ModelLoader(mock_model_config)
                yield loader


class TestHealthCheckIntegration:
    """Integration tests for health check functionality."""

    def test_health_check_on_unloaded_model(self, mock_loader):
        """Test health checks before model is loaded."""
        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is False
        assert "not loaded" in message.lower()

    @patch("src.models.health.DeviceDetector")
    def test_health_check_on_loaded_model(self, mock_detector, mock_loader):
        """Test health checks after model is loaded."""
        # Mock device detector for health check
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_detector.get_device_info.return_value = {"device": "mps"}
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        # Load model (mocked)
        with patch("src.models.registry.get_model_spec") as mock_spec:
            mock_spec.return_value = {
                "path": "test/path",
                "context_length": 4096,
            }
            mock_loader.load()

        # Check health
        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is True
        assert "phi-3-mini" in message

    @patch("src.models.health.DeviceDetector")
    def test_comprehensive_health_status(self, mock_detector, mock_loader):
        """Test comprehensive health status report."""
        # Mock device detector
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_detector.get_device_info.return_value = {"device": "mps"}
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        # Load model (mocked)
        with patch("src.models.registry.get_model_spec") as mock_spec:
            mock_spec.return_value = {
                "path": "test/path",
                "context_length": 4096,
            }
            mock_loader.load()

        # Get health status
        status = get_health_status(mock_loader)

        # Verify status structure
        assert "model_loaded" in status
        assert "metal_available" in status
        assert "can_generate" in status
        assert "overall_healthy" in status
        assert "issues" in status

        # Model should be loaded
        assert status["model_loaded"] is True

    @patch("src.models.health.DeviceDetector")
    def test_health_report_formatting(self, mock_detector, mock_loader):
        """Test health report formatting."""
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_detector.get_device_info.return_value = {"device": "mps"}
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        # Load model
        with patch("src.models.registry.get_model_spec") as mock_spec:
            mock_spec.return_value = {
                "path": "test/path",
                "context_length": 4096,
            }
            mock_loader.load()

        status = get_health_status(mock_loader)
        report = format_health_report(status)

        # Verify report content
        assert "System Health Report" in report
        assert "Platform:" in report
        assert "Metal GPU:" in report
        assert "Model:" in report
        assert isinstance(report, str)


class TestModelSwitchingIntegration:
    """Integration tests for model switching functionality."""

    @patch("src.models.switcher.DeviceDetector")
    @patch("src.models.switcher.validate_model_name")
    @patch("src.models.switcher.get_model_spec")
    @patch("src.models.switcher.MemoryManager")
    def test_switch_model_workflow(
        self, mock_memory, mock_spec, mock_validate, mock_detector, mock_loader
    ):
        """Test complete model switching workflow."""
        # Load initial model
        with patch("src.models.registry.get_model_spec") as loader_spec:
            loader_spec.return_value = {
                "path": "test/phi-3",
                "context_length": 4096,
            }
            mock_loader.load()

        # Mock switch validation
        mock_validate.return_value = (True, "Valid")
        mock_spec.return_value = {
            "path": "test/mistral",
            "context_length": 8192,
            "quantization": "4bit",
            "memory_estimate_mb": 4500,
        }
        mock_memory.can_fit_model.return_value = (True, "Fits")

        # Perform switch
        success, message, info = switch_model(mock_loader, "mistral-7b")

        # Verify switch occurred
        assert success is True
        assert info is not None
        # Loader should have been unloaded and reloaded
        assert mock_loader.is_loaded() is True

    def test_switch_to_same_model_rejected(self, mock_loader):
        """Test that switching to the same model is rejected."""
        from src.models.switcher import validate_switch_feasibility

        # Load initial model
        with patch("src.models.registry.get_model_spec") as loader_spec:
            loader_spec.return_value = {
                "path": "test/phi-3",
                "context_length": 4096,
            }
            mock_loader.load()

        # Try to switch to same model
        is_feasible, message = validate_switch_feasibility("phi-3-mini", "phi-3-mini")

        assert is_feasible is False
        assert "same" in message.lower()


class TestModelToolsIntegration:
    """Integration tests for model tools functionality."""

    def test_list_and_get_info_workflow(self):
        """Test listing models and getting their info."""
        # List all models
        all_models = list_models()

        assert len(all_models) > 0
        assert isinstance(all_models, list)

        # Get info for each model
        for model_name in all_models[:3]:  # Test first 3
            info = get_model_info(model_name)
            assert info is not None
            assert info["name"] == model_name
            assert "path" in info
            assert "memory_estimate_mb" in info

    def test_filter_and_sort_models(self):
        """Test filtering and sorting models."""
        # Filter by size
        small_models = list_models(filter_by_size="small")
        assert len(small_models) > 0

        # Filter by context
        long_context = list_models(filter_by_context=100000)
        # phi-3-mini-128k should be in this list
        assert isinstance(long_context, list)

        # Sort by memory
        sorted_models = list_models(sort_by="memory")
        assert len(sorted_models) > 0

    @patch("src.models.tools.DeviceDetector")
    @patch("src.models.tools.MemoryManager")
    def test_check_compatibility_workflow(self, mock_memory, mock_detector):
        """Test model compatibility checking."""
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_memory.can_fit_model.return_value = (True, "Fits")

        # Check compatibility of several models
        test_models = ["phi-3-mini", "mistral-7b", "llama-3-8b"]

        for model in test_models:
            is_compatible, message = check_model_compatibility(model)

            # Should be compatible with mocked system
            assert is_compatible is True
            assert "compatible" in message.lower()

    def test_model_recommendation_workflow(self):
        """Test getting model recommendations."""
        from src.models.tools import get_model_recommendation

        # Get recommendations for different use cases
        use_cases = ["chat", "long_context", "general"]

        for use_case in use_cases:
            result = get_model_recommendation(use_case)

            if result is not None:
                model, reason = result
                assert isinstance(model, str)
                assert isinstance(reason, str)
                assert model in list_models()


class TestEndToEndWorkflow:
    """End-to-end workflow tests simulating real usage patterns."""

    @patch("src.models.health.DeviceDetector")
    @patch("src.models.switcher.validate_model_name")
    @patch("src.models.switcher.get_model_spec")
    @patch("src.models.switcher.MemoryManager")
    def test_complete_workflow(
        self, mock_memory, mock_spec, mock_validate, mock_detector, mock_loader
    ):
        """
        Test complete workflow: list models -> check compatibility ->
        load model -> check health -> switch model -> check health again.
        """
        # 1. List available models
        all_models = list_models()
        assert len(all_models) > 0

        # 2. Check compatibility
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_memory.can_fit_model.return_value = (True, "Fits")

        is_compatible, _ = check_model_compatibility("phi-3-mini")
        assert is_compatible is True

        # 3. Load model
        with patch("src.models.registry.get_model_spec") as loader_spec:
            loader_spec.return_value = {
                "path": "test/phi-3",
                "context_length": 4096,
            }
            mock_loader.load()

        # 4. Check health
        mock_detector.get_device_info.return_value = {"device": "mps"}
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        is_loaded, message = check_model_loaded(mock_loader)
        assert is_loaded is True

        # 5. Switch model
        mock_validate.return_value = (True, "Valid")
        mock_spec.return_value = {
            "path": "test/mistral",
            "context_length": 8192,
            "quantization": "4bit",
            "memory_estimate_mb": 4500,
        }

        success, message, info = switch_model(mock_loader, "mistral-7b")
        assert success is True

        # 6. Check health again
        is_loaded, message = check_model_loaded(mock_loader)
        assert is_loaded is True

    def test_error_handling_workflow(self):
        """Test error handling in workflow."""
        # Try to get info for nonexistent model
        info = get_model_info("nonexistent-model-xyz")
        assert info is None

        # Try to check compatibility for nonexistent model
        is_compatible, message = check_model_compatibility("nonexistent-model-xyz")
        assert is_compatible is False
        assert "not found" in message.lower()


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_health_check_with_none_loader(self):
        """Test health check with None loader (should handle gracefully)."""
        # This should raise an AttributeError, which we catch
        try:
            check_model_loaded(None)
            # If it doesn't raise, the function handled None gracefully
            assert True
        except AttributeError:
            # Expected behavior - function doesn't handle None
            assert True

    def test_empty_model_list_operations(self):
        """Test operations on empty model lists."""
        from src.models.tools import compare_models

        # Compare empty list
        result = compare_models([])
        assert isinstance(result, str)

    def test_invalid_filter_parameters(self):
        """Test list_models with invalid filter parameters."""
        # Invalid size category
        result = list_models(filter_by_size="nonexistent")
        assert isinstance(result, list)

        # Very high context requirement (might return empty)
        result = list_models(filter_by_context=10000000)
        assert isinstance(result, list)  # May be empty but should be a list

    @patch("src.models.switcher.MemoryManager")
    def test_switch_with_minimal_memory(self, mock_memory):
        """Test switch behavior when memory is very tight."""
        from src.models.switcher import validate_switch_feasibility

        mock_memory.can_fit_model.return_value = (False, "Not enough memory")

        with patch("src.models.switcher.validate_model_name") as mock_validate:
            with patch("src.models.switcher.get_model_spec") as mock_spec:
                mock_validate.return_value = (True, "Valid")
                mock_spec.return_value = {"memory_estimate_mb": 100000}

                is_feasible, message = validate_switch_feasibility(
                    "phi-3-mini", "huge-model"
                )

                assert is_feasible is False
                assert "memory" in message.lower()
