"""
Shared pytest fixtures for GPT-OSS CLI Chat testing.

This module provides common fixtures used across unit and integration tests,
including mock objects, configuration fixtures, and test data.
"""

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

import pytest

# Add src to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_model_config():
    """Create a mock ModelConfig for testing."""
    config = Mock()
    config.model_name = "gpt-j-20b"
    config.model_path = None
    config.quantization = "int4"
    config.device = "auto"
    config.tensor_parallel_size = 1
    config.gpu_memory_utilization = 0.90
    config.max_model_len = 4096
    config.lazy_load = True
    config.warmup = True
    config.trust_remote_code = True
    config.validate = Mock()
    return config


@pytest.fixture
def mock_inference_config():
    """Create a mock InferenceConfig for testing."""
    config = Mock()
    config.temperature = 0.7
    config.top_p = 0.9
    config.top_k = 50
    config.repetition_penalty = 1.0
    config.max_tokens = 512
    config.streaming = True
    config.stop_sequences = None
    config.validate = Mock()
    return config


@pytest.fixture
def mock_cli_config():
    """Create a mock CLIConfig for testing."""
    config = Mock()
    config.colorize = True
    config.show_tokens = True
    config.show_latency = True
    config.verbose = False
    config.auto_save = False
    config.history_file = ".chat_history.json"
    config.enable_autocomplete = True
    config.multiline_input = False
    return config


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_model_info():
    """Create a mock ModelInfo object."""
    info = Mock()
    info.model_name = "gpt-j-20b"
    info.model_path = "EleutherAI/gpt-j-20b"
    info.quantization = "int4"
    info.device = "cuda"
    info.memory_usage_mb = 7000
    info.context_length = 2048
    info.is_loaded = True
    info.load_time_ms = 5000
    return info


@pytest.fixture
def mock_model_loader(mock_model_config, mock_model_info):
    """Create a mock ModelLoader."""
    loader = Mock()
    loader.config = mock_model_config
    loader.is_loaded = Mock(return_value=True)
    loader.load = Mock(return_value=mock_model_info)
    loader.unload = Mock()
    loader.get_info = Mock(return_value=mock_model_info)
    loader.health_check = Mock(return_value=True)
    loader.switch_model = Mock(return_value=mock_model_info)
    return loader


@pytest.fixture
def mock_vllm_model():
    """Create a mock vLLM model."""
    model = MagicMock()
    model.generate = Mock(return_value=["Generated text response"])
    return model


# =============================================================================
# Inference Fixtures
# =============================================================================

@pytest.fixture
def mock_generation_result():
    """Create a mock GenerationResult."""
    result = Mock()
    result.text = "This is a generated response."
    result.tokens_generated = 10
    result.latency_ms = 150
    result.finish_reason = "stop"
    return result


@pytest.fixture
def mock_inference_engine(mock_model_loader, mock_inference_config, mock_generation_result):
    """Create a mock InferenceEngine."""
    engine = Mock()
    engine.model_loader = mock_model_loader
    engine.config = mock_inference_config
    engine.generate = Mock(return_value=mock_generation_result)
    engine.generate_stream = Mock(return_value=iter(["This", " is", " a", " test"]))
    engine.cancel_generation = Mock()
    return engine


# =============================================================================
# Conversation Fixtures
# =============================================================================

@pytest.fixture
def mock_message():
    """Create a mock Message."""
    message = Mock()
    message.role = "user"
    message.content = "Hello, how are you?"
    message.timestamp = "2025-01-01T12:00:00"
    message.tokens = 5
    return message


@pytest.fixture
def mock_conversation_manager():
    """Create a mock ConversationManager."""
    manager = Mock()
    manager.max_context_tokens = 4096
    manager.add_message = Mock(return_value=Mock())
    manager.get_history = Mock(return_value=[])
    manager.format_prompt = Mock(return_value="<|user|>\nTest prompt\n<|assistant|>\n")
    manager.clear = Mock()
    manager.save = Mock()
    manager.load = Mock(return_value=Mock())
    manager.get_token_count = Mock(return_value=0)
    return manager


# =============================================================================
# Device Detection Fixtures
# =============================================================================

@pytest.fixture
def mock_device_detector():
    """Create a mock DeviceDetector."""
    detector = Mock()
    detector.detect = Mock(return_value="cuda")
    detector.is_cuda_available = Mock(return_value=True)
    detector.is_mps_available = Mock(return_value=False)
    detector.get_cuda_device_count = Mock(return_value=1)
    detector.get_cuda_memory = Mock(return_value=(2000, 8000))
    return detector


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    manager = Mock()
    manager.get_model_memory_usage = Mock(return_value=7000)
    manager.estimate_memory_requirement = Mock(return_value=7000)
    manager.can_fit_model = Mock(return_value=True)
    return manager


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Clear all GPT-OSS related environment variables."""
    env_vars = [
        "MODEL_NAME", "QUANTIZATION", "DEVICE", "TENSOR_PARALLEL_SIZE",
        "GPU_MEMORY_UTIL", "TEMPERATURE", "TOP_P", "MAX_TOKENS",
        "STREAMING", "COLORIZE", "SHOW_TOKENS", "VERBOSE"
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_env_vars(monkeypatch):
    """Set up sample environment variables for testing."""
    monkeypatch.setenv("MODEL_NAME", "gpt-neox-20b")
    monkeypatch.setenv("QUANTIZATION", "int8")
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("TEMPERATURE", "0.8")
    monkeypatch.setenv("TOP_P", "0.95")
    monkeypatch.setenv("STREAMING", "false")


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_history_file(tmp_path):
    """Create a temporary conversation history file path."""
    return tmp_path / ".chat_history.json"


@pytest.fixture
def sample_conversation_data():
    """Provide sample conversation data for testing."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "timestamp": "2025-01-01T12:00:00",
                "tokens": 6
            },
            {
                "role": "user",
                "content": "What is Python?",
                "timestamp": "2025-01-01T12:01:00",
                "tokens": 4
            },
            {
                "role": "assistant",
                "content": "Python is a high-level programming language.",
                "timestamp": "2025-01-01T12:01:05",
                "tokens": 10
            }
        ]
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: marks tests that require GPU access"
    )
    config.addinivalue_line(
        "markers",
        "requires_model: marks tests that require model files"
    )


# =============================================================================
# Test Helpers
# =============================================================================

class MockVLLMOutput:
    """Mock vLLM generation output."""

    def __init__(self, text: str, tokens: int = 10):
        self.text = text
        self.tokens = tokens
        self.outputs = [Mock(text=text, token_ids=list(range(tokens)))]


@pytest.fixture
def mock_vllm_output():
    """Create a mock vLLM output object."""
    return MockVLLMOutput("This is a test response.", tokens=6)
