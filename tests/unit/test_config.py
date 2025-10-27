"""
Unit tests for configuration classes.

Tests ModelConfig, InferenceConfig, and CLIConfig validation and loading.
"""

import pytest


# =============================================================================
# ModelConfig Tests
# =============================================================================

class TestModelConfig:
    """Test suite for ModelConfig class."""

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_defaults(self):
        """Test that ModelConfig default values are valid."""
        from src.config import ModelConfig

        config = ModelConfig()
        config.validate()

        # Verify defaults
        assert config.model_name == "gpt-j-20b"
        assert config.quantization == "int4"
        assert config.device == "auto"
        assert config.tensor_parallel_size == 1
        assert 0.5 <= config.gpu_memory_utilization <= 1.0
        assert 512 <= config.max_model_len <= 32768

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_from_env(self, sample_env_vars):
        """Test loading ModelConfig from environment variables."""
        from src.config import ModelConfig

        config = ModelConfig.from_env()

        assert config.model_name == "gpt-neox-20b"
        assert config.quantization == "int8"
        assert config.device == "cuda"

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_validation_gpu_memory(self):
        """Test ModelConfig validates GPU memory utilization."""
        from src.config import ModelConfig

        # Too low
        config = ModelConfig(gpu_memory_utilization=0.3)
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            config.validate()

        # Too high
        config = ModelConfig(gpu_memory_utilization=1.5)
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            config.validate()

        # Valid
        config = ModelConfig(gpu_memory_utilization=0.85)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_validation_tensor_parallel(self):
        """Test ModelConfig validates tensor parallel size."""
        from src.config import ModelConfig

        # Invalid
        config = ModelConfig(tensor_parallel_size=0)
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            config.validate()

        config = ModelConfig(tensor_parallel_size=-1)
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            config.validate()

        # Valid
        config = ModelConfig(tensor_parallel_size=2)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_validation_max_model_len(self):
        """Test ModelConfig validates max model length."""
        from src.config import ModelConfig

        # Too small
        config = ModelConfig(max_model_len=256)
        with pytest.raises(ValueError, match="max_model_len"):
            config.validate()

        # Too large
        config = ModelConfig(max_model_len=50000)
        with pytest.raises(ValueError, match="max_model_len"):
            config.validate()

        # Valid
        config = ModelConfig(max_model_len=2048)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_quantization_options(self):
        """Test ModelConfig accepts valid quantization options."""
        from src.config import ModelConfig

        valid_options = ["int4", "int8", "fp16", "none"]

        for option in valid_options:
            config = ModelConfig(quantization=option)
            config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_device_options(self):
        """Test ModelConfig accepts valid device options."""
        from src.config import ModelConfig

        valid_devices = ["cuda", "mps", "cpu", "auto"]

        for device in valid_devices:
            config = ModelConfig(device=device)
            config.validate()  # Should not raise


# =============================================================================
# InferenceConfig Tests
# =============================================================================

class TestInferenceConfig:
    """Test suite for InferenceConfig class."""

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_defaults(self):
        """Test that InferenceConfig default values are valid."""
        from src.config import InferenceConfig

        config = InferenceConfig()
        config.validate()

        # Verify defaults
        assert 0.0 <= config.temperature <= 2.0
        assert 0.0 <= config.top_p <= 1.0
        assert config.max_tokens >= 1
        assert isinstance(config.streaming, bool)

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_from_env(self, sample_env_vars):
        """Test loading InferenceConfig from environment variables."""
        from src.config import InferenceConfig

        config = InferenceConfig.from_env()

        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.streaming is False

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_validation_temperature(self):
        """Test InferenceConfig validates temperature."""
        from src.config import InferenceConfig

        # Too low
        config = InferenceConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

        # Too high
        config = InferenceConfig(temperature=2.5)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

        # Valid range
        for temp in [0.0, 0.7, 1.0, 2.0]:
            config = InferenceConfig(temperature=temp)
            config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_validation_top_p(self):
        """Test InferenceConfig validates top_p."""
        from src.config import InferenceConfig

        # Invalid
        config = InferenceConfig(top_p=-0.1)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

        config = InferenceConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

        # Valid
        config = InferenceConfig(top_p=0.9)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_validation_max_tokens(self):
        """Test InferenceConfig validates max_tokens."""
        from src.config import InferenceConfig

        # Too small
        config = InferenceConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

        # Too large
        config = InferenceConfig(max_tokens=5000)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

        # Valid
        config = InferenceConfig(max_tokens=512)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_stop_sequences(self):
        """Test InferenceConfig handles stop sequences."""
        from src.config import InferenceConfig

        config = InferenceConfig(stop_sequences=["<|endoftext|>", "\n\n"])
        config.validate()

        assert len(config.stop_sequences) == 2
        assert "<|endoftext|>" in config.stop_sequences


# =============================================================================
# CLIConfig Tests
# =============================================================================

class TestCLIConfig:
    """Test suite for CLIConfig class."""

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement CLIConfig")
    def test_cli_config_defaults(self):
        """Test that CLIConfig default values are valid."""
        from src.config import CLIConfig

        config = CLIConfig()

        # Verify defaults
        assert isinstance(config.colorize, bool)
        assert isinstance(config.show_tokens, bool)
        assert isinstance(config.show_latency, bool)
        assert isinstance(config.verbose, bool)
        assert isinstance(config.history_file, str)

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement CLIConfig")
    def test_cli_config_from_env(self, sample_env_vars, monkeypatch):
        """Test loading CLIConfig from environment variables."""
        from src.config import CLIConfig

        monkeypatch.setenv("COLORIZE", "false")
        monkeypatch.setenv("SHOW_TOKENS", "false")
        monkeypatch.setenv("VERBOSE", "true")

        config = CLIConfig.from_env()

        assert config.colorize is False
        assert config.show_tokens is False
        assert config.verbose is True

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement CLIConfig")
    def test_cli_config_boolean_parsing(self, monkeypatch):
        """Test CLIConfig correctly parses boolean environment variables."""
        from src.config import CLIConfig

        # Test various boolean string representations
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("COLORIZE", env_value)
            config = CLIConfig.from_env()
            assert config.colorize is expected


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfigIntegration:
    """Integration tests for configuration system."""

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement all configs")
    def test_all_configs_can_be_imported(self):
        """Test that all config classes can be imported together."""
        from src.config import ModelConfig, InferenceConfig, CLIConfig

        assert ModelConfig is not None
        assert InferenceConfig is not None
        assert CLIConfig is not None

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement all configs")
    def test_all_configs_can_be_instantiated(self):
        """Test that all config classes can be instantiated with defaults."""
        from src.config import ModelConfig, InferenceConfig, CLIConfig

        model_config = ModelConfig()
        inference_config = InferenceConfig()
        cli_config = CLIConfig()

        model_config.validate()
        inference_config.validate()

        assert model_config is not None
        assert inference_config is not None
        assert cli_config is not None

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement all configs")
    def test_configs_work_together(self, sample_env_vars):
        """Test that all configs can be loaded from environment together."""
        from src.config import ModelConfig, InferenceConfig, CLIConfig

        model_config = ModelConfig.from_env()
        inference_config = InferenceConfig.from_env()
        cli_config = CLIConfig.from_env()

        # All should load without errors
        model_config.validate()
        inference_config.validate()

        # Verify some values
        assert model_config.model_name == "gpt-neox-20b"
        assert inference_config.temperature == 0.8
        assert cli_config is not None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestConfigEdgeCases:
    """Test edge cases and error handling in config classes."""

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
    def test_model_config_empty_string_handling(self):
        """Test ModelConfig handles empty strings appropriately."""
        from src.config import ModelConfig

        # Empty model name should use default
        config = ModelConfig(model_name="")
        # Depending on implementation, this might use default or raise error

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement InferenceConfig")
    def test_inference_config_boundary_values(self):
        """Test InferenceConfig with boundary values."""
        from src.config import InferenceConfig

        # Test minimum valid values
        config = InferenceConfig(temperature=0.0, top_p=0.0, max_tokens=1)
        config.validate()  # Should not raise

        # Test maximum valid values
        config = InferenceConfig(temperature=2.0, top_p=1.0, max_tokens=4096)
        config.validate()  # Should not raise

    @pytest.mark.skip(reason="Waiting for infrastructure-agent to implement configs")
    def test_config_with_none_values(self):
        """Test how configs handle None values."""
        from src.config import ModelConfig

        # Optional fields should accept None
        config = ModelConfig(model_path=None, stop_sequences=None)
        config.validate()  # Should not raise
