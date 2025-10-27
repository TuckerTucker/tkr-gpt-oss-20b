# GPT-OSS CLI Chat - Testing Infrastructure

Comprehensive testing infrastructure for the GPT-OSS CLI Chat project.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                    # Unit tests (mocked dependencies)
│   ├── test_config.py       # Configuration validation tests
│   ├── test_model_loader.py # Model loading and device management
│   ├── test_inference.py    # Inference engine and generation
│   └── test_conversation.py # Conversation management
└── integration/             # Integration tests (real components)
    └── (Wave 2 deliverables)
```

## Running Tests

### Run all unit tests
```bash
pytest tests/unit/ -v
```

### Run specific test file
```bash
pytest tests/unit/test_config.py -v
```

### Run with coverage
```bash
pytest tests/unit/ --cov=src --cov-report=html
```

### Skip slow tests
```bash
pytest tests/unit/ -m "not slow"
```

## Test Markers

Tests are marked with custom markers:

- `@pytest.mark.slow` - Tests that take significant time (real model loading)
- `@pytest.mark.integration` - Integration tests requiring multiple components
- `@pytest.mark.requires_gpu` - Tests requiring GPU access
- `@pytest.mark.requires_model` - Tests requiring model files

## Current Status

**Wave 1**: All unit tests are implemented with `@pytest.mark.skip` decorators. Tests will automatically enable as other agents implement their modules.

### Test Coverage

- **Config Tests** (19 tests): ModelConfig, InferenceConfig, CLIConfig validation
- **Model Loader Tests** (23 tests): Device detection, memory management, model loading
- **Inference Tests** (24 tests): Text generation, streaming, metrics tracking
- **Conversation Tests** (28 tests): History management, prompt formatting, persistence

**Total**: 94 test cases ready for validation

## Fixtures

The `conftest.py` file provides comprehensive fixtures for all components:

### Configuration Fixtures
- `mock_model_config` - ModelConfig with defaults
- `mock_inference_config` - InferenceConfig with defaults
- `mock_cli_config` - CLIConfig with defaults

### Model Fixtures
- `mock_model_loader` - Mocked ModelLoader
- `mock_model_info` - Mocked ModelInfo
- `mock_vllm_model` - Mocked vLLM model

### Inference Fixtures
- `mock_inference_engine` - Mocked InferenceEngine
- `mock_generation_result` - Mocked GenerationResult

### Conversation Fixtures
- `mock_conversation_manager` - Mocked ConversationManager
- `mock_message` - Mocked Message

### Device Fixtures
- `mock_device_detector` - Mocked DeviceDetector
- `mock_memory_manager` - Mocked MemoryManager

### Environment Fixtures
- `clean_env` - Clears environment variables
- `sample_env_vars` - Sets sample environment variables
- `temp_history_file` - Temporary file for conversation persistence

## Integration with Other Agents

Tests are designed to validate integration contracts:

- **infrastructure-agent** → Config validation
- **model-agent** → Model loading and device management
- **inference-agent** → Text generation and streaming
- **conversation-agent** → Conversation history and prompts
- **cli-agent** → Ready for integration tests (Wave 2)

## Removing Skip Decorators

As agents implement their modules, remove the `@pytest.mark.skip` decorator:

```python
# Before (skipped)
@pytest.mark.skip(reason="Waiting for infrastructure-agent to implement ModelConfig")
def test_model_config_defaults():
    ...

# After (active)
def test_model_config_defaults():
    ...
```

Tests will then run automatically and validate the implementation.

## Wave 2 Additions

Integration tests will be added in Wave 2:

- E2E workflow tests (CLI → generation → output)
- Real model loading tests (marked `@pytest.mark.slow`)
- Multi-component integration tests
- Performance regression tests

## Benchmarking

Performance benchmarks are in `benchmarks/`:

```bash
# Run latency benchmark
python benchmarks/latency_benchmark.py

# With custom settings
python benchmarks/latency_benchmark.py --model gpt-j-20b --prompts 20
```

## Validation Scripts

Helper scripts in `scripts/`:

```bash
# Validate environment setup
python scripts/validate_setup.py

# Run Wave 1 validation gate
./scripts/validate_wave1.sh
```

## Contributing

When adding new tests:

1. Use descriptive test names: `test_<component>_<behavior>`
2. Add docstrings explaining what is tested
3. Use appropriate fixtures from `conftest.py`
4. Mark slow tests with `@pytest.mark.slow`
5. Mock external dependencies (vLLM, file system, etc.)

## Testing Philosophy

- **Independence**: Tests don't depend on other agents' implementations
- **Mocking**: External dependencies are mocked for fast, reliable tests
- **Coverage**: Test happy paths, edge cases, and error conditions
- **Documentation**: Clear test names and docstrings explain intent
- **Progressive**: Tests enable automatically as code is implemented
