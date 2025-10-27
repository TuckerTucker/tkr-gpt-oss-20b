# Testing Quick Reference

## Quick Commands

### Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_config.py -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=html

# Skip slow tests
pytest tests/unit/ -m "not slow"

# Stop on first failure
pytest tests/unit/ -x

# Show print output
pytest tests/unit/ -s

# Verbose with full diff
pytest tests/unit/ -vv
```

### Validation

```bash
# Check environment setup
python scripts/validate_setup.py

# Verbose setup check
python scripts/validate_setup.py --verbose

# Wave 1 validation gate
./scripts/validate_wave1.sh

# Verbose gate check
./scripts/validate_wave1.sh --verbose
```

### Benchmarking

```bash
# Run benchmark with defaults
python benchmarks/latency_benchmark.py

# Custom benchmark
python benchmarks/latency_benchmark.py \
  --model gpt-j-20b \
  --quantization int4 \
  --prompts 20 \
  --tokens 100

# Save results to JSON
python benchmarks/latency_benchmark.py --output results.json
```

## Test Files Map

| File | Module | Tests | Coverage |
|------|--------|-------|----------|
| `test_config.py` | Config validation | 19 | ModelConfig, InferenceConfig, CLIConfig |
| `test_model_loader.py` | Model management | 23 | ModelLoader, DeviceDetector, MemoryManager |
| `test_inference.py` | Text generation | 24 | InferenceEngine, streaming, metrics |
| `test_conversation.py` | Conversation | 28 | ConversationManager, templates, persistence |

**Total: 94 test cases**

## Common Test Patterns

### Testing with fixtures
```python
def test_something(mock_model_config, mock_model_loader):
    # Use fixtures directly
    assert mock_model_config.model_name == "gpt-j-20b"
    assert mock_model_loader.is_loaded() is True
```

### Testing with environment variables
```python
def test_env_loading(sample_env_vars):
    # sample_env_vars sets test env vars
    config = ModelConfig.from_env()
    assert config.model_name == "gpt-neox-20b"
```

### Testing with temporary files
```python
def test_save(temp_history_file):
    # temp_history_file is a Path to temp file
    manager.save(str(temp_history_file))
    assert temp_history_file.exists()
```

### Testing exceptions
```python
def test_validation_error():
    config = ModelConfig(gpu_memory_utilization=1.5)
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        config.validate()
```

### Mocking external calls
```python
@patch('vllm.LLM.generate')
def test_generation(mock_generate):
    mock_generate.return_value = [Mock(text="Response")]
    result = engine.generate("prompt")
    assert result.text == "Response"
```

## Removing Skip Decorators

When a module is implemented, remove the skip decorator:

```python
# Before (skipped)
@pytest.mark.skip(reason="Waiting for infrastructure-agent")
def test_model_config_defaults():
    from src.config import ModelConfig
    config = ModelConfig()
    config.validate()

# After (active) - just remove the decorator line
def test_model_config_defaults():
    from src.config import ModelConfig
    config = ModelConfig()
    config.validate()
```

## Custom Markers

Tests use these markers:

```python
@pytest.mark.slow         # Long-running tests (real models)
@pytest.mark.integration  # Multi-component tests
@pytest.mark.requires_gpu # Needs GPU access
@pytest.mark.requires_model # Needs model files
```

Run specific markers:
```bash
pytest -m "slow"              # Only slow tests
pytest -m "not slow"          # Skip slow tests
pytest -m "integration"       # Only integration tests
```

## Fixture Categories

### Config Fixtures
- `mock_model_config`
- `mock_inference_config`
- `mock_cli_config`

### Model Fixtures
- `mock_model_loader`
- `mock_model_info`
- `mock_vllm_model`

### Inference Fixtures
- `mock_inference_engine`
- `mock_generation_result`

### Conversation Fixtures
- `mock_conversation_manager`
- `mock_message`

### Device Fixtures
- `mock_device_detector`
- `mock_memory_manager`

### Environment Fixtures
- `clean_env` - Clears env vars
- `sample_env_vars` - Sets test env vars
- `temp_history_file` - Temp file path
- `sample_conversation_data` - Sample JSON data

## File Locations

```
tests/
├── conftest.py              # All fixtures defined here
├── unit/
│   ├── test_config.py       # Config tests
│   ├── test_model_loader.py # Model tests
│   ├── test_inference.py    # Inference tests
│   └── test_conversation.py # Conversation tests
└── integration/             # Wave 2

benchmarks/
└── latency_benchmark.py     # Performance benchmark

scripts/
├── validate_setup.py        # Environment check
└── validate_wave1.sh       # Wave 1 gate check
```

## Troubleshooting

### Tests are all skipped
✓ **Expected** - Tests skip until modules are implemented

### Import errors
✓ **Expected** - Modules not yet implemented by other agents

### pytest not found
```bash
pip install pytest
```

### Missing fixtures
Check `tests/conftest.py` - all fixtures defined there

### Syntax errors
```bash
python3 -m py_compile tests/unit/test_*.py
```

## Integration with Agents

| Agent | Tests Ready | Remove Skip When |
|-------|-------------|------------------|
| infrastructure-agent | test_config.py | src/config/ implemented |
| model-agent | test_model_loader.py | src/models/, src/devices/ implemented |
| inference-agent | test_inference.py | src/inference/, src/sampling/ implemented |
| conversation-agent | test_conversation.py | src/conversation/, src/prompts/ implemented |
| cli-agent | Wave 2 integration tests | src/cli/ implemented |

## Success Indicators

✓ All Python files compile
✓ pytest runs without import errors
✓ Tests skip gracefully with clear messages
✓ validate_setup.py runs
✓ validate_wave1.sh reports expected warnings
✓ Benchmarks are executable

## Wave 1 Status

**Current**: All test infrastructure complete
**Status**: ✓ Ready for validation
**Blockers**: None (independent design)
**Next**: Wait for other agents to implement modules

## Help & Resources

- Test documentation: `tests/README.md`
- Full details: `tests/TESTING_INFRASTRUCTURE.md`
- Status log: `.context-kit/orchestration/gpt-oss-cli-chat/status/testing-agent.log`
- Integration contracts: `.context-kit/orchestration/gpt-oss-cli-chat/integration-contracts/`
