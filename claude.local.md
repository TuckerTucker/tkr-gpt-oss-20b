# Claude Code Persistent Context - tkr-gpt-oss-20b

This file contains persistent context for Claude Code AI sessions to maintain project understanding across conversations.

## System Prompt

You are working on tkr-gpt-oss-20b, an MLX-optimized CLI chatbot for Apple Silicon that enables running 20B+ parameter models (GPT-J, GPT-NeoX, Phi-3) locally on M1/M2/M3 Macs.

## Project Overview

- **Type**: CLI chatbot application
- **Platform**: Apple Silicon (M1/M2/M3) exclusive
- **Framework**: Python 3.10+ with MLX for hardware-accelerated inference
- **Models**: GPT-J-20B, GPT-NeoX-20B, Phi-3-Mini with int4/int8 quantization
- **Status**: Alpha (v0.1.0)

## Key Architecture Decisions

1. **Modular Design**: 8 main modules with clear separation of concerns
   - `cli/`: REPL interface and terminal UI using Rich and prompt-toolkit
   - `config/`: Multi-layered configuration (env vars → CLI args → defaults)
   - `conversation/`: History management with JSON persistence
   - `devices/`: MLX/Metal device detection and optimization
   - `inference/`: Streaming text generation engine with error recovery
   - `models/`: Model lifecycle management with lazy loading
   - `prompts/`: Template system for prompt engineering
   - `sampling/`: Generation parameter management

2. **Streaming Architecture**: Token-by-token generation for responsive UX
   - Real-time streaming from MLX inference
   - Metrics tracking (tokens/second, latency)
   - Error recovery with automatic retries

3. **Configuration System**: Three-layer config hierarchy
   - Environment variables (.env file)
   - Command line arguments (override env)
   - Sensible defaults (fallback)

4. **Memory Optimization**: Quantization for running large models
   - int4: Best memory/quality tradeoff (recommended)
   - int8: Better quality, more memory
   - Enables 20B models on 16GB+ RAM

## Critical File Locations

- Entry point: `src/main.py`
- Configuration: `pyproject.toml`, `requirements.txt`, `.env.example`
- Tests: `tests/unit/`, `tests/integration/`
- Benchmarks: `benchmarks/latency_benchmark.py`, `benchmarks/integration_benchmark.py`
- Validation: `scripts/validate_setup.py`, `scripts/validate_wave*.sh`

## Common Development Tasks

### Setup and Run
```bash
# Setup virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run chatbot
python src/main.py

# With options
python src/main.py --model phi-3-mini --temperature 0.9 --verbose
```

### Testing and Quality
```bash
# Run tests
pytest tests/ -v

# Linting and formatting
ruff check src/ tests/
black src/ tests/

# Type checking
mypy src/
```

### Benchmarking
```bash
# Latency benchmark
python benchmarks/latency_benchmark.py

# Integration benchmark
python benchmarks/integration_benchmark.py
```

## Important Patterns and Conventions

1. **Type Safety**: All code uses type annotations with mypy checking
2. **Error Recovery**: Automatic retry logic in inference engine
3. **Conversation Management**: Full history with save/load/search capabilities
4. **Device Detection**: Automatic MLX/Metal optimization for Apple Silicon
5. **Model Registry**: Centralized model management with health checks

## Available CLI Commands

- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/history` - Display conversation
- `/save [file]` - Save conversation to file
- `/load [file]` - Load conversation from file
- `/info` - Show model and session information
- `/exit` or `/quit` - Exit the chat

## Code Quality Standards

- **Formatting**: Black with 100 character line length
- **Linting**: Ruff with pycodestyle, pyflakes, isort, bugbear, comprehensions
- **Type Checking**: mypy with strict mode enabled
- **Testing**: pytest with unit and integration tests
- **Python Version**: 3.10+ required

## Dependencies

### Production
- mlx >= 0.4.0 (Apple MLX framework)
- mlx-lm >= 0.8.0 (MLX language models)
- transformers >= 4.35.0 (HuggingFace)
- tokenizers >= 0.15.0 (Fast tokenizers)
- rich >= 13.7.0 (Terminal UI)
- prompt-toolkit >= 3.0.43 (Input handling)
- python-dotenv >= 1.0.0 (Environment management)
- pyyaml >= 6.0.1 (YAML parsing)

### Development
- pytest >= 7.4.3
- pytest-asyncio >= 0.21.1
- black >= 23.12.0
- mypy >= 1.7.1
- ruff >= 0.1.8

## _context-kit.yml

```yaml
# Project configuration for AI agents - tkr-gpt-oss-20b
# Optimized per Repo-Context Format v1.0 - YAML 1.2 compliant
meta:
  kit: tkr-context-kit
  fmt: 1
  type: cli-chatbot
  desc: "MLX-optimized CLI chatbot for Apple Silicon with 20B+ parameter models (GPT-J, GPT-NeoX, Phi-3)"
  ver: "0.1.0"
  author: "Tucker <tucker@tkr-projects.com>"
  ts: "2025-10-27T10:00:00Z"
  status: alpha
  entry: "src/main.py"
  stack: "Python 3.10+ + MLX + HuggingFace Transformers + Rich CLI"
  cmds: ["python src/main.py", "pytest tests/ -v", "python benchmarks/latency_benchmark.py"]
  achievements: ["Apple Silicon optimized inference", "Streaming token generation", "Conversation history management", "Multi-model support", "Advanced error recovery"]

# Dependencies with MLX focus
deps: &deps
  py: &py-deps
    prod:
      # Core MLX dependencies
      mlx: &mlx {id: null, v: ">=0.4.0", desc: "Apple MLX framework"}
      mlx-lm: &mlx-lm {id: null, v: ">=0.8.0", desc: "MLX language models"}
      # Model and tokenization
      transformers: {id: null, v: ">=4.35.0", desc: "HuggingFace transformers"}
      tokenizers: {id: null, v: ">=0.15.0", desc: "Fast tokenizers"}
      # CLI interface
      rich: {id: null, v: ">=13.7.0", desc: "Rich terminal output"}
      prompt-toolkit: {id: null, v: ">=3.0.43", desc: "Advanced input handling"}
      # Configuration and utilities
      python-dotenv: {id: null, v: ">=1.0.0", desc: "Environment management"}
      pyyaml: {id: null, v: ">=6.0.1", desc: "YAML parsing"}
    dev:
      pytest: &pytest {id: null, v: ">=7.4.3", desc: "Testing framework"}
      pytest-asyncio: {id: null, v: ">=0.21.1", desc: "Async testing"}
      black: {id: null, v: ">=23.12.0", desc: "Code formatting"}
      mypy: {id: null, v: ">=1.7.1", desc: "Type checking"}
      ruff: {id: null, v: ">=0.1.8", desc: "Linting"}

# Directory structure (focusing on project files, excluding venv/cache)
struct:
  _: {n: 75, t: {py: 75}}

  src:
    _: {n: 43, t: {py: 43}}
    main.py: "Entry point and CLI argument parsing"

    cli:
      _: {n: 6, files: [commands.py, display.py, input.py, output.py, repl.py]}
      desc: "REPL interface and terminal UI"

    config:
      _: {n: 4, files: [cli_config.py, inference_config.py, model_config.py]}
      desc: "Configuration management from env/CLI"

    conversation:
      _: {n: 8, files: [context.py, export.py, history.py, persistence.py, search.py, stats.py, tokens.py]}
      desc: "Conversation history and context management"

    devices:
      _: {n: 2, files: [detector.py]}
      desc: "MLX/Metal device detection"

    inference:
      _: {n: 8, files: [batch.py, engine.py, exceptions.py, metrics.py, optimization.py, recovery.py, streaming.py]}
      desc: "Text generation engine with streaming"

    models:
      _: {n: 8, files: [exceptions.py, health.py, loader.py, memory.py, registry.py, switcher.py, tools.py]}
      desc: "Model loading and management"

    prompts:
      _: {n: 3, files: [presets.py, templates.py]}
      desc: "Prompt templates and presets"

    sampling:
      _: {n: 2, files: [params.py]}
      desc: "Sampling parameter management"

  tests:
    _: {n: 20, t: {py: 20}}
    unit: {n: 9, desc: "Unit tests for individual components"}
    integration: {n: 10, desc: "End-to-end workflow tests"}
    conftest.py: "Pytest configuration and fixtures"

  benchmarks:
    _: {n: 2, t: {py: 2}}
    files: [latency_benchmark.py, integration_benchmark.py]
    desc: "Performance benchmarking suite"

  scripts:
    _: {n: 3, t: {py: 1, sh: 2}}
    files: [validate_setup.py, validate_wave1.sh, validate_wave2.sh]
    desc: "Validation and setup scripts"

  examples:
    _: {n: 3}
    desc: "Example usage scripts"

  docs:
    _: {n: 1, t: {md: 1}}
    desc: "Documentation"

  config:
    pyproject.toml: "Python project metadata and tool config"
    requirements.txt: "Production dependencies"
    .env.example: "Environment configuration template"

# Architecture patterns
arch:
  stack:
    lang: "Python 3.10+"
    runtime: "Apple Silicon (M1/M2/M3) with MLX"
    framework: "MLX + mlx-lm for inference"
    models: ["GPT-J-20B", "GPT-NeoX-20B", "Phi-3-Mini"]
    ui: "Rich terminal + prompt-toolkit REPL"
    persistence: "JSON conversation history"

  patterns: &arch-patterns
    - "Modular architecture with clear separation of concerns"
    - "Configuration via environment variables and CLI args"
    - "Streaming token generation for responsive UI"
    - "Conversation history with save/load/search"
    - "Error recovery with automatic retry logic"
    - "Device detection and MLX optimization"
    - "Model registry with dynamic loading"
    - "Token counting and context window management"
    - "Type-safe configuration with dataclasses"

  modules:
    cli:
      responsibility: "User interface and command handling"
      features: ["REPL loop", "Rich formatting", "Command autocompletion", "Multi-line input"]

    config:
      responsibility: "Configuration management"
      features: ["Environment variable loading", "CLI argument parsing", "Config merging", "Validation"]

    conversation:
      responsibility: "Conversation state management"
      features: ["History tracking", "Save/load", "Search", "Token counting", "Context window management"]

    devices:
      responsibility: "Hardware detection"
      features: ["MLX device detection", "Metal acceleration", "Memory monitoring"]

    inference:
      responsibility: "Text generation"
      features: ["Streaming generation", "Batch processing", "Error recovery", "Metrics collection", "Optimization"]

    models:
      responsibility: "Model lifecycle management"
      features: ["Lazy loading", "Model switching", "Health checks", "Memory management", "Registry"]

    prompts:
      responsibility: "Prompt engineering"
      features: ["Template system", "Presets", "System prompts"]

    sampling:
      responsibility: "Generation parameters"
      features: ["Temperature", "Top-p/top-k", "Repetition penalty", "Stop sequences"]

  data_flow:
    - "User input → CLI → Conversation → Inference → Model → Streaming → CLI → User"
    - "Config loading: .env → Environment → CLI args → Merged config"
    - "Model loading: Registry → Loader → Health check → Ready"
    - "History: Persistence → History → Context → Tokens → Save"

# Operations and workflows
ops:
  development:
    setup: "python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    run: "python src/main.py [options]"
    test: "pytest tests/ -v"
    lint: "ruff check src/ tests/"
    format: "black src/ tests/"
    typecheck: "mypy src/"

  usage:
    basic: "python src/main.py"
    model_override: "python src/main.py --model phi-3-mini"
    temperature_adjust: "python src/main.py --temperature 0.9"
    no_streaming: "python src/main.py --no-streaming"
    verbose: "python src/main.py --verbose"

  commands:
    help: "/help - Show available commands"
    clear: "/clear - Clear conversation history"
    history: "/history - Display conversation"
    save: "/save [file] - Save conversation"
    load: "/load [file] - Load conversation"
    info: "/info - Show model and session info"
    exit: "/exit or /quit - Exit chat"

  config:
    env_file: ".env for persistent settings"
    cli_args: "Command line arguments override env"
    defaults: "Sensible defaults in config modules"

  models:
    phi_3_mini: {size: "3.8B", ram: "8GB+", context: "4K", desc: "Fast, capable, good for most tasks"}
    gpt_j_20b: {size: "20B", ram: "24GB+", context: "2K", desc: "Very capable, needs high-end Mac"}
    gpt_neox_20b: {size: "20B", ram: "24GB+", context: "2K", desc: "Excellent quality, high memory"}

  quantization:
    int4: "Best memory/quality tradeoff (recommended)"
    int8: "Better quality, more memory"
    fp16: "Full precision, highest memory"
    none: "No quantization"

# Testing patterns
testing:
  unit:
    config: "Configuration loading and validation"
    conversation: "History management and context tracking"
    inference: "Generation engine and streaming"
    models: "Model loading and health checks"
    cli: "Command parsing and execution"

  integration:
    e2e_workflow: "Complete chat workflow with mock model"
    config_to_model: "Config propagation to model loading"
    model_to_inference: "Model integration with inference engine"
    inference_streaming: "Streaming generation pipeline"
    conversation_to_cli: "History integration with CLI"

  benchmarks:
    latency: "Token generation speed and load times"
    integration: "Full workflow performance"

# Key features and capabilities
features:
  inference:
    streaming: "Token-by-token generation for responsive UI"
    batch: "Batch processing support"
    optimization: "MLX hardware acceleration"
    recovery: "Automatic error recovery with retries"
    metrics: "Generation time and tokens/second tracking"

  conversation:
    history: "Full conversation tracking"
    persistence: "JSON save/load"
    search: "Search within conversation"
    context: "Context window management"
    tokens: "Token counting and limits"
    export: "Export to various formats"

  cli:
    repl: "Interactive read-eval-print loop"
    rich_output: "Colorized and formatted output"
    commands: "Built-in slash commands"
    multiline: "Multi-line message input"
    autocomplete: "Command completion"

  config:
    env_vars: "Environment variable configuration"
    cli_args: "Command line argument overrides"
    validation: "Type-safe config validation"
    defaults: "Sensible default values"

  models:
    registry: "Centralized model registry"
    lazy_load: "Load models on first use"
    switching: "Hot-swap between models"
    health: "Model health checks"
    memory: "Memory usage monitoring"

# Semantic context for AI consumption
semantic:
  ~apple_silicon: "Optimized for M1/M2/M3 Macs using MLX framework"
  ~large_models: "Run 20B parameter models locally with quantization"
  ~streaming: "Token-by-token streaming for natural conversation flow"
  ~conversation_management: "Save/load/search conversation history"
  ~modular_architecture: "Clean separation between CLI, config, inference, models"
  ~error_recovery: "Robust error handling with automatic retries"
  ~performance: "Hardware-accelerated inference with metrics tracking"
  ~flexibility: "Configure via env vars, config files, or CLI args"
  ~type_safety: "Type annotations and dataclass configs throughout"
  ~testing: "Comprehensive unit and integration test coverage"

# Architecture evolution notes
notes:
  - "VERSION: 0.1.0 - Initial alpha release"
  - "PLATFORM: Apple Silicon exclusive (M1/M2/M3 required)"
  - "MODELS: Support for GPT-J-20B, GPT-NeoX-20B, Phi-3-Mini"
  - "QUANTIZATION: Int4/Int8 for memory efficiency"
  - "STREAMING: Real-time token generation for responsive UX"
  - "CONFIGURATION: Multi-layered config system (env → CLI → defaults)"
  - "TESTING: 29 tests covering unit and integration scenarios"
  - "BENCHMARKS: Performance testing for latency and throughput"
  - "ERROR_HANDLING: Automatic recovery with retry logic"
  - "CONVERSATION: Full history management with persistence"
  - "CLI_UX: Rich terminal UI with command completion"
  - "MEMORY_MANAGEMENT: Model health monitoring and optimization"
  - "MODULAR_DESIGN: 8 main modules with clear responsibilities"
  - "TYPE_SAFETY: Full type annotations with mypy checking"
  - "CODE_QUALITY: Black formatting + Ruff linting"
```
