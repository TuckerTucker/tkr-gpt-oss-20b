# GPT-OSS CLI Chat

Interactive command-line chatbot powered by large language models (20B+ parameters) optimized for Apple Silicon using MLX.

Talk to powerful open-source models like GPT-J, GPT-NeoX, and Phi-3 directly from your terminal with hardware-accelerated inference on M1/M2/M3 Macs.

## Features

- **Apple Silicon Optimized**: Hardware-accelerated inference using MLX for maximum performance on M-series chips
- **20B+ Parameter Models**: Run GPT-J-20B, GPT-NeoX-20B, and other large models locally
- **Memory Efficient**: Int4/Int8 quantization enables 20B models on 16GB+ RAM
- **Streaming Responses**: Token-by-token streaming for natural conversation flow
- **Rich Terminal UI**: Colorized output, command completion, and formatted messages
- **Conversation Management**: Save/load chat history, context tracking, token counting
- **Flexible Configuration**: Environment variables, config files, and CLI arguments
- **Harmony Integration**: Multi-channel output with reasoning traces for enhanced transparency

## Quick Start

### Prerequisites

- **Hardware**: Apple Silicon Mac (M1, M2, M3, or newer)
- **OS**: macOS 12.0+ (Monterey or later)
- **RAM**: 16GB minimum (32GB recommended for 20B models)
- **Python**: 3.10 or newer

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tkr-gpt-oss-20b
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create configuration (optional)**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

5. **Run the chatbot**
   ```bash
   python src/main.py
   ```

The first run will download the model from HuggingFace (this may take several minutes depending on your connection).

## Usage

### Basic Usage

Start a chat session with default settings:

```bash
python src/main.py
```

### Command Line Options

Override configuration via command line arguments:

```bash
# Use a different model
python src/main.py --model phi-3-mini

# Adjust sampling temperature (0.0 = focused, 2.0 = creative)
python src/main.py --temperature 0.9

# Generate longer responses
python src/main.py --max-tokens 1024

# Disable streaming for batch-style output
python src/main.py --no-streaming

# Enable debug logging
python src/main.py --verbose
```

### Available Commands

Inside the chat interface, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/history` | Display conversation history |
| `/save [file]` | Save conversation to file |
| `/load [file]` | Load conversation from file |
| `/info` | Show model and session information |
| `/exit` or `/quit` | Exit the chat (or press Ctrl+C) |

### Example Session

```
$ python src/main.py --model phi-3-mini

╔══════════════════════════════════════════════════════════════════╗
║              GPT-OSS CLI Chat - MLX Optimized                     ║
║                  Model: phi-3-mini (4bit)                        ║
╚══════════════════════════════════════════════════════════════════╝

Type a message to begin chatting.
Commands: /help, /history, /save, /load, /clear, /info, /exit

> Hello! What can you help me with?

I'm a helpful AI assistant! I can help you with:
- Answering questions and providing information
- Writing and editing content
- Explaining concepts and ideas
- Code assistance and debugging
- Creative writing and brainstorming
- And much more!

What would you like to know?

> /info

═══════════════════════════════════════════════════════════════
Model Information
═══════════════════════════════════════════════════════════════
Model: phi-3-mini
Device: mps
Quantization: int4
Memory Usage: 2,456 MB
Context Length: 4,096 tokens

Session Statistics
═══════════════════════════════════════════════════════════════
Messages: 4
Total Tokens: 287
Conversation Started: 2025-10-27 10:30:00
═══════════════════════════════════════════════════════════════
```

## Harmony Integration

GPT-OSS CLI now supports **Harmony multi-channel format**, enabling models to output multiple information streams simultaneously:

- **Analysis Channel**: Internal reasoning and chain-of-thought (developer visibility)
- **Commentary Channel**: Meta-commentary about actions taken
- **Final Channel**: User-facing response content

### Why Use Harmony?

- **Enhanced Quality**: Models "think out loud" leading to better responses
- **Transparency**: Access reasoning traces for debugging and understanding
- **Configurable**: Adjust reasoning levels (low/medium/high) based on task needs
- **Safety**: Analysis channel filtered out by default - never shown to end users

### Quick Start with Harmony

```bash
# Default mode (Harmony enabled, medium reasoning)
python src/main.py

# Show reasoning traces while chatting
python src/main.py --show-reasoning

# Adjust reasoning level for different tasks
python src/main.py --reasoning-level low    # Fast (extraction, simple queries)
python src/main.py --reasoning-level medium # Balanced (chat, default)
python src/main.py --reasoning-level high   # Thorough (research, analysis)

# Capture reasoning for later analysis
python src/main.py --capture-reasoning

# Combine options for debugging
python src/main.py --reasoning-level high --show-reasoning --capture-reasoning
```

### Reasoning Levels

| Level  | Use Case | Speed | Quality | Token Usage |
|--------|----------|-------|---------|-------------|
| **low** | Extraction, simple Q&A | Fast | Good | +5-10% |
| **medium** | Chat, summarization | Balanced | Better | +10-20% |
| **high** | Research, complex analysis | Slower | Best | +20-40% |

### Example with Reasoning

```bash
$ python src/main.py --show-reasoning

> Explain quantum entanglement

[Reasoning Trace]
User asks about quantum entanglement - complex physics topic. I should:
1. Provide a clear definition
2. Use an accessible analogy
3. Mention key properties (non-locality, measurement correlation)
4. Note practical implications

[Response]
Quantum entanglement is a phenomenon where two particles become connected
such that the quantum state of one particle instantly influences the other,
regardless of distance. Think of it like having two magic coins: when you
flip one and it lands on heads, the other instantly becomes tails, even if
it's on the other side of the universe. This "spooky action at a distance"
(as Einstein called it) is real and has been experimentally verified.
It's the basis for emerging technologies like quantum computing and
quantum cryptography.
```

### Programmatic Usage

```python
from src.inference.engine import InferenceEngine
from src.config.inference_config import InferenceConfig, ReasoningLevel

# Configure Harmony
config = InferenceConfig(
    use_harmony_format=True,
    reasoning_level=ReasoningLevel.HIGH,
    capture_reasoning=True
)

# Generate with reasoning
engine = InferenceEngine(model_loader, config)
result = engine.generate(prompt, params)

# Access results
print(result.text)       # Final channel (user-safe)
print(result.reasoning)  # Analysis channel (developer-only)
print(result.commentary) # Commentary channel (if present)
```

### Learn More

- **User Guide**: [docs/harmony_integration.md](docs/harmony_integration.md) - Comprehensive guide to Harmony features
- **Migration Guide**: [docs/migration_to_harmony.md](docs/migration_to_harmony.md) - Upgrading from legacy format
- **Examples**: [examples/harmony_prompts.py](examples/harmony_prompts.py) - Working code examples
- **Safety Note**: Always use `result.text` for end users - analysis channel is not user-safe

## Configuration

Configure the chatbot using environment variables in a `.env` file:

```bash
# Model Settings
MODEL_NAME=phi-3-mini          # Model to load
QUANTIZATION=int4              # Memory optimization (int4, int8, fp16, none)
DEVICE=auto                    # Device selection (auto, mps, cuda, cpu)
MAX_MODEL_LEN=4096            # Context window size

# Inference Settings
TEMPERATURE=0.7                # Sampling temperature (0.0-2.0)
TOP_P=0.9                      # Nucleus sampling threshold
MAX_TOKENS=512                 # Max tokens per response
STREAMING=true                 # Enable token streaming

# CLI Settings
COLORIZE=true                  # Colored output
SHOW_TOKENS=true              # Display token counts
SHOW_LATENCY=true             # Display generation time
AUTO_SAVE=false               # Auto-save on exit
HISTORY_FILE=.chat_history.json
```

See `.env.example` for complete configuration options.

## Supported Models

Models optimized for Apple Silicon via MLX:

| Model | Size | RAM Required | Context | Description |
|-------|------|--------------|---------|-------------|
| **phi-3-mini** | 3.8B | 8GB+ | 4K | Fast, capable, good for most tasks |
| **gpt-j-20b** | 20B | 24GB+ | 2K | Very capable, needs high-end Mac |
| **gpt-neox-20b** | 20B | 24GB+ | 2K | Excellent quality, high memory |

*RAM requirements shown are for Int4 quantization. Int8 and fp16 require more memory.*

### Choosing a Model

- **For general use on 16GB Macs**: `phi-3-mini`
- **For best quality (32GB+ RAM)**: `gpt-j-20b` or `gpt-neox-20b`
- **For testing**: Use `--test` flag to run without loading a model

## Architecture

```
tkr-gpt-oss-20b/
├── src/
│   ├── main.py              # Entry point (this Wave)
│   ├── config/              # Configuration system
│   │   ├── model_config.py
│   │   ├── inference_config.py
│   │   └── cli_config.py
│   ├── models/              # Model loading and management
│   │   ├── loader.py
│   │   ├── registry.py
│   │   └── memory.py
│   ├── devices/             # Device detection (MLX/Metal)
│   ├── inference/           # Text generation engine
│   │   ├── engine.py
│   │   ├── streaming.py
│   │   └── metrics.py
│   ├── conversation/        # History management
│   │   ├── history.py
│   │   ├── context.py
│   │   └── persistence.py
│   ├── prompts/            # Prompt templates
│   └── cli/                # REPL and commands
│       ├── repl.py
│       ├── commands.py
│       ├── input.py
│       └── display.py
├── tests/                  # Test suite
├── benchmarks/            # Performance benchmarks
├── examples/              # Example scripts
└── docs/                  # Documentation
```

## Performance

Expected performance on Apple Silicon (M2 Max, 32GB):

| Model | Quantization | Load Time | Tokens/Second | Memory |
|-------|--------------|-----------|---------------|--------|
| phi-3-mini | int4 | ~10s | ~45 | 2.5GB |
| phi-3-mini | int8 | ~15s | ~35 | 4.5GB |
| gpt-j-20b | int4 | ~45s | ~12 | 12GB |
| gpt-j-20b | int8 | ~60s | ~8 | 22GB |

*Performance varies by hardware. M1/M2 Pro with 16GB RAM can run smaller models efficiently.*

## Troubleshooting

### Model won't load

```bash
# Check you're on Apple Silicon
uname -m  # Should show "arm64"

# Verify MLX is installed
python -c "import mlx; print(mlx.__version__)"

# Try test mode (no model loading)
python src/main.py --test
```

### Out of memory errors

- Use int4 quantization: `--quantization int4`
- Try smaller model: `--model phi-3-mini`
- Reduce context window: `--max-model-len 2048`
- Close other applications

### Slow performance

- Ensure Metal acceleration is working (check Activity Monitor)
- Use int4 quantization for best speed/quality tradeoff
- Reduce `--max-tokens` for faster responses
- Check that you're not running on battery power (throttling)

See `docs/TROUBLESHOOTING.md` for detailed troubleshooting.

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_config.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

### Benchmarks

```bash
# Run latency benchmark
python benchmarks/latency_benchmark.py

# Run throughput benchmark
python benchmarks/throughput_benchmark.py
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## License

[License information here]

## Acknowledgments

- **MLX**: Apple's machine learning framework for Apple Silicon
- **mlx-lm**: Language model inference library built on MLX
- **HuggingFace**: Model hosting and transformers library
- **Rich**: Beautiful terminal formatting
- **Prompt Toolkit**: Enhanced input handling

## Citation

```bibtex
@software{gpt_oss_cli_chat,
  title = {GPT-OSS CLI Chat},
  author = {Tucker},
  year = {2025},
  description = {MLX-optimized chatbot for Apple Silicon}
}
```

---

**Made with ❤️ for Apple Silicon**