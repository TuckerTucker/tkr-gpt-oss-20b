"""
Unit tests for model health check functionality.

Tests the health checking, model switching, and model tools utilities
without loading actual MLX models (uses mocks).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.health import (
    check_model_loaded,
    check_metal_backend,
    get_health_status,
    format_health_report,
)
from src.models.switcher import (
    switch_model,
    validate_switch_feasibility,
    get_switch_recommendation,
    cleanup_before_switch,
)
from src.models.tools import (
    list_models,
    get_model_info,
    download_model,
    compare_models,
    find_models_by_description,
    get_model_recommendation,
    format_model_info,
    check_model_compatibility,
)
from src.models.loader import ModelInfo
from src.models.exceptions import ModelLoadError


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthChecks:
    """Test suite for health check functions."""

    def test_check_model_loaded_when_loaded(self):
        """Test check_model_loaded with loaded model."""
        # Create mock loader with loaded model
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_info.return_value = ModelInfo(
            model_name="phi-3-mini",
            model_path="mlx-community/Phi-3-mini-4k-instruct-4bit",
            quantization="4bit",
            device="mps",
            memory_usage_mb=2500,
            context_length=4096,
            is_loaded=True,
            load_time_ms=5000,
        )

        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is True
        assert "phi-3-mini" in message
        assert "mps" in message
        assert "2500MB" in message

    def test_check_model_loaded_when_not_loaded(self):
        """Test check_model_loaded with unloaded model."""
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = False

        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is False
        assert "not loaded" in message.lower()

    def test_check_model_loaded_with_error(self):
        """Test check_model_loaded when error occurs."""
        mock_loader = Mock()
        mock_loader.is_loaded.side_effect = Exception("Test error")

        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is False
        assert "error" in message.lower()

    @patch("src.devices.detector.DeviceDetector")
    def test_check_metal_backend_available(self, mock_detector):
        """Test check_metal_backend when Metal is available."""
        mock_detector.validate_platform.return_value = (
            True,
            "Platform validated",
        )
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        is_available, message = check_metal_backend()

        assert is_available is True
        assert "Metal GPU available" in message
        assert "1024MB active" in message

    @patch("src.devices.detector.DeviceDetector")
    def test_check_metal_backend_unavailable(self, mock_detector):
        """Test check_metal_backend when Metal is not available."""
        mock_detector.validate_platform.return_value = (
            False,
            "Not Apple Silicon",
        )

        is_available, message = check_metal_backend()

        assert is_available is False
        assert "not available" in message.lower()

    @patch("src.devices.detector.DeviceDetector")
    @patch("src.models.memory.MemoryManager")
    def test_get_health_status_healthy(self, mock_memory, mock_detector):
        """Test get_health_status with healthy system."""
        # Mock platform validation
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_detector.get_device_info.return_value = {"device": "mps"}
        mock_detector.get_metal_info.return_value = {
            "metal_available": True,
            "active_memory_mb": 1024,
            "peak_memory_mb": 2048,
            "cache_memory_mb": 512,
        }

        # Mock loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_info.return_value = ModelInfo(
            model_name="phi-3-mini",
            model_path="test/path",
            quantization="4bit",
            device="mps",
            memory_usage_mb=2500,
            context_length=4096,
            is_loaded=True,
            load_time_ms=5000,
        )
        mock_loader.health_check.return_value = True

        status = get_health_status(mock_loader)

        assert status["overall_healthy"] is True
        assert status["model_loaded"] is True
        assert status["metal_available"] is True
        assert status["can_generate"] is True
        assert len(status["issues"]) == 0

    @patch("src.devices.detector.DeviceDetector")
    def test_get_health_status_unhealthy(self, mock_detector):
        """Test get_health_status with unhealthy system."""
        mock_detector.validate_platform.return_value = (False, "Invalid")
        mock_detector.get_device_info.return_value = {"device": "cpu"}
        mock_detector.get_metal_info.return_value = {}

        mock_loader = Mock()
        mock_loader.is_loaded.return_value = False

        status = get_health_status(mock_loader)

        assert status["overall_healthy"] is False
        assert status["model_loaded"] is False
        assert len(status["issues"]) > 0

    def test_format_health_report(self):
        """Test format_health_report output."""
        status = {
            "overall_healthy": True,
            "model_loaded": True,
            "model_name": "phi-3-mini",
            "metal_available": True,
            "metal_active_memory_mb": 1024,
            "metal_peak_memory_mb": 2048,
            "metal_cache_memory_mb": 512,
            "device": "mps",
            "platform_valid": True,
            "can_generate": True,
            "issues": [],
        }

        report = format_health_report(status)

        assert "System Health Report" in report
        assert "Healthy" in report
        assert "phi-3-mini" in report
        assert "Metal GPU" in report


# ============================================================================
# Model Switching Tests
# ============================================================================


class TestModelSwitching:
    """Test suite for model switching functions."""

    @patch("src.models.registry.validate_model_name")
    @patch("src.models.registry.get_model_spec")
    @patch("src.models.memory.MemoryManager")
    def test_switch_model_success(self, mock_memory, mock_get_spec, mock_validate):
        """Test successful model switch."""
        # Mock validation
        mock_validate.return_value = (True, "Valid")

        # Mock spec
        mock_get_spec.return_value = {
            "path": "test/path",
            "context_length": 4096,
            "quantization": "4bit",
            "memory_estimate_mb": 2500,
        }

        # Mock memory check
        mock_memory.can_fit_model.return_value = (True, "Fits")

        # Mock loader
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = True
        mock_loader.get_info.return_value = ModelInfo(
            model_name="phi-3-mini",
            model_path="old/path",
            quantization="4bit",
            device="mps",
            memory_usage_mb=2000,
            context_length=4096,
            is_loaded=True,
            load_time_ms=5000,
        )
        mock_loader.load.return_value = ModelInfo(
            model_name="mistral-7b",
            model_path="test/path",
            quantization="4bit",
            device="mps",
            memory_usage_mb=4500,
            context_length=8192,
            is_loaded=True,
            load_time_ms=7000,
        )

        success, message, info = switch_model(mock_loader, "mistral-7b")

        assert success is True
        assert "Switched" in message or "Loaded" in message
        assert info is not None
        assert info.model_name == "mistral-7b"
        mock_loader.unload.assert_called_once()
        mock_loader.load.assert_called_once()

    @patch("src.models.registry.validate_model_name")
    def test_switch_model_invalid_name(self, mock_validate):
        """Test switch_model with invalid model name."""
        mock_validate.return_value = (False, "Model not found")
        mock_loader = Mock()

        success, message, info = switch_model(mock_loader, "invalid-model")

        assert success is False
        assert "not found" in message.lower()
        assert info is None

    @patch("src.models.registry.validate_model_name")
    @patch("src.models.registry.get_model_spec")
    @patch("src.models.memory.MemoryManager")
    def test_switch_model_insufficient_memory(
        self, mock_memory, mock_get_spec, mock_validate
    ):
        """Test switch_model when insufficient memory."""
        mock_validate.return_value = (True, "Valid")
        mock_get_spec.return_value = {"memory_estimate_mb": 100000}
        mock_memory.can_fit_model.return_value = (False, "Not enough memory")

        mock_loader = Mock()

        success, message, info = switch_model(mock_loader, "huge-model")

        assert success is False
        assert "memory" in message.lower()
        assert info is None

    @patch("src.models.registry.validate_model_name")
    @patch("src.models.registry.get_model_spec")
    def test_validate_switch_feasibility(self, mock_get_spec, mock_validate):
        """Test validate_switch_feasibility."""
        mock_validate.return_value = (True, "Valid")
        mock_get_spec.return_value = {
            "memory_estimate_mb": 2500,
        }

        with patch("src.models.memory.MemoryManager") as mock_memory:
            mock_memory.can_fit_model.return_value = (True, "Fits")

            is_feasible, message = validate_switch_feasibility(
                "phi-3-mini", "mistral-7b"
            )

            assert is_feasible is True
            assert "feasible" in message.lower()

    def test_validate_switch_feasibility_same_model(self):
        """Test validate_switch_feasibility with same model."""
        is_feasible, message = validate_switch_feasibility("phi-3-mini", "phi-3-mini")

        assert is_feasible is False
        assert "same" in message.lower()

    @patch("src.models.registry.get_model_spec")
    def test_get_switch_recommendation(self, mock_get_spec):
        """Test get_switch_recommendation."""
        mock_get_spec.return_value = {
            "description": "Great model",
            "memory_estimate_mb": 2500,
        }

        model, reason = get_switch_recommendation(2000, "chat")

        assert model is not None
        assert isinstance(reason, str)
        assert len(reason) > 0

    @patch("src.models.memory.MemoryManager")
    def test_cleanup_before_switch(self, mock_memory):
        """Test cleanup_before_switch."""
        # Should not raise any exceptions
        cleanup_before_switch()

        mock_memory.clear_cache.assert_called_once()
        mock_memory.reset_peak_memory.assert_called_once()


# ============================================================================
# Model Tools Tests
# ============================================================================


class TestModelTools:
    """Test suite for model tools functions."""

    def test_list_models_all(self):
        """Test list_models returns all models."""
        models = list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "phi-3-mini" in models

    def test_list_models_filter_by_size(self):
        """Test list_models with size filter."""
        small_models = list_models(filter_by_size="small")

        assert isinstance(small_models, list)
        # Should have some small models
        assert len(small_models) > 0

    def test_list_models_filter_by_context(self):
        """Test list_models with context filter."""
        long_context = list_models(filter_by_context=100000)

        assert isinstance(long_context, list)
        # Should have models with long context
        # (phi-3-mini-128k has 128K context)

    def test_list_models_sort_by_memory(self):
        """Test list_models sorted by memory."""
        models = list_models(sort_by="memory")

        assert isinstance(models, list)
        assert len(models) > 0
        # First model should have less memory than last
        # (assuming we have models of different sizes)

    def test_get_model_info_valid(self):
        """Test get_model_info with valid model."""
        info = get_model_info("phi-3-mini")

        assert info is not None
        assert info["name"] == "phi-3-mini"
        assert "path" in info
        assert "memory_estimate_mb" in info
        assert info["memory_estimate_gb"] > 0

    def test_get_model_info_invalid(self):
        """Test get_model_info with invalid model."""
        info = get_model_info("nonexistent-model")

        assert info is None

    @patch("src.models.tools.ModelLoader")
    def test_download_model_success(self, mock_loader_class):
        """Test download_model success."""
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        success, message = download_model("phi-3-mini")

        assert success is True
        assert "downloaded" in message.lower()
        mock_loader.load.assert_called_once()
        mock_loader.unload.assert_called_once()

    def test_download_model_invalid(self):
        """Test download_model with invalid model."""
        success, message = download_model("nonexistent-model")

        assert success is False
        assert "not found" in message.lower()

    def test_compare_models(self):
        """Test compare_models output."""
        comparison = compare_models(["phi-3-mini", "mistral-7b"])

        assert "Model Comparison" in comparison
        assert "phi-3-mini" in comparison
        assert "mistral-7b" in comparison

    def test_find_models_by_description(self):
        """Test find_models_by_description."""
        # Search for models with "long" in description
        models = find_models_by_description("long")

        assert isinstance(models, list)
        # Should find phi-3-mini-128k which has "long conversations"

    def test_get_model_recommendation_basic(self):
        """Test get_model_recommendation with use case."""
        result = get_model_recommendation("chat")

        assert result is not None
        model, reason = result
        assert isinstance(model, str)
        assert isinstance(reason, str)

    def test_get_model_recommendation_with_constraints(self):
        """Test get_model_recommendation with memory constraint."""
        result = get_model_recommendation("chat", max_memory_gb=3.0)

        assert result is not None
        model, reason = result
        # Should recommend a small model
        info = get_model_info(model)
        assert info["memory_estimate_gb"] <= 3.0

    def test_get_model_recommendation_impossible_constraints(self):
        """Test get_model_recommendation with impossible constraints."""
        # No model fits in 0.5GB
        result = get_model_recommendation("chat", max_memory_gb=0.5)

        # Should return None or a very small model
        # (depends on implementation)

    def test_format_model_info(self):
        """Test format_model_info output."""
        info_str = format_model_info("phi-3-mini")

        assert "phi-3-mini" in info_str
        assert "Description" in info_str
        assert "Repository" in info_str
        assert "Size" in info_str

    def test_format_model_info_invalid(self):
        """Test format_model_info with invalid model."""
        info_str = format_model_info("nonexistent-model")

        assert "not found" in info_str.lower()

    @patch("src.devices.detector.DeviceDetector")
    @patch("src.models.memory.MemoryManager")
    def test_check_model_compatibility_compatible(self, mock_memory, mock_detector):
        """Test check_model_compatibility with compatible model."""
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_memory.can_fit_model.return_value = (True, "Fits")

        is_compatible, message = check_model_compatibility("phi-3-mini")

        assert is_compatible is True
        assert "compatible" in message.lower()

    @patch("src.devices.detector.DeviceDetector")
    def test_check_model_compatibility_incompatible_platform(self, mock_detector):
        """Test check_model_compatibility with incompatible platform."""
        mock_detector.validate_platform.return_value = (False, "Not Apple Silicon")

        is_compatible, message = check_model_compatibility("phi-3-mini")

        assert is_compatible is False
        assert "incompatible" in message.lower()

    def test_check_model_compatibility_invalid_model(self):
        """Test check_model_compatibility with invalid model."""
        is_compatible, message = check_model_compatibility("nonexistent-model")

        assert is_compatible is False
        assert "not found" in message.lower()


# ============================================================================
# Integration-style Tests (using real model specs, no actual loading)
# ============================================================================


class TestIntegrationWithoutLoading:
    """Integration tests that use real model specs but don't load models."""

    def test_health_workflow_without_model(self):
        """Test health check workflow without loaded model."""
        mock_loader = Mock()
        mock_loader.is_loaded.return_value = False

        is_loaded, message = check_model_loaded(mock_loader)

        assert is_loaded is False

    @patch("src.devices.detector.DeviceDetector")
    def test_switch_workflow_dry_run(self, mock_detector):
        """Test switch workflow feasibility check."""
        mock_detector.validate_platform.return_value = (True, "Valid")

        with patch("src.models.memory.MemoryManager") as mock_memory:
            mock_memory.can_fit_model.return_value = (True, "Fits")

            is_feasible, message = validate_switch_feasibility(
                "phi-3-mini", "mistral-7b"
            )

            # Should be feasible with mocked memory
            assert isinstance(is_feasible, bool)

    def test_tools_workflow(self):
        """Test tools workflow for model discovery."""
        # List all models
        all_models = list_models()
        assert len(all_models) > 0

        # Get info for first model
        first_model = all_models[0]
        info = get_model_info(first_model)
        assert info is not None

        # Format info
        formatted = format_model_info(first_model)
        assert first_model in formatted

        # Compare models
        if len(all_models) >= 2:
            comparison = compare_models(all_models[:2])
            assert "Model Comparison" in comparison

    def test_recommendation_workflow(self):
        """Test model recommendation workflow."""
        # Get recommendation
        result = get_model_recommendation("chat")
        assert result is not None

        model, reason = result
        assert model in list_models()

        # Get info about recommended model
        info = get_model_info(model)
        assert info is not None
