# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with GPT-OSS CLI Chat.

## Table of Contents

- [Installation Issues](#installation-issues)
- [MLX and Metal Issues](#mlx-and-metal-issues)
- [Model Loading Issues](#model-loading-issues)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Inference Errors](#inference-errors)
- [Configuration Issues](#configuration-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### MLX installation fails

**Problem**: `pip install mlx` fails with compilation errors

**Solutions**:

1. **Verify Apple Silicon**
   ```bash
   uname -m  # Should output "arm64"
   ```
   MLX only works on Apple Silicon Macs (M1, M2, M3, etc.)

2. **Update pip and build tools**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. **Check Python version**
   ```bash
   python --version  # Should be 3.10 or newer
   ```

4. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

### mlx-lm not found

**Problem**: `ModuleNotFoundError: No module named 'mlx_lm'`

**Solution**:
```bash
# Install mlx-lm specifically
pip install mlx-lm

# Or reinstall all requirements
pip install -r requirements.txt
```

### Transformers version conflicts

**Problem**: Incompatible transformers version

**Solution**:
```bash
# Install specific compatible versions
pip install transformers>=4.35.0 tokenizers>=0.15.0

# Or force reinstall
pip install --force-reinstall transformers tokenizers
```

---

## MLX and Metal Issues

### Metal backend not available

**Problem**: Model runs on CPU instead of GPU

**Diagnosis**:
```bash
# Check Metal availability
python -c "import mlx.core as mx; print(mx.default_device())"
# Should show "Device(gpu, 0)" not "Device(cpu)"
```

**Solutions**:

1. **Check macOS version**
   - MLX requires macOS 12.0 (Monterey) or newer
   ```bash
   sw_vers
   ```

2. **Restart Metal drivers**
   ```bash
   # Sometimes Metal needs a restart
   sudo killall -HUP mds
   ```

3. **Check Activity Monitor**
   - Open Activity Monitor
   - Go to Window > GPU History
   - Verify GPU is active when running inference

### Metal allocation errors

**Problem**: `RuntimeError: Metal buffer allocation failed`

**Solutions**:

1. **Close other GPU-intensive apps**
   - Close games, video editors, multiple browsers

2. **Reduce GPU memory usage**
   ```bash
   # Reduce memory utilization in .env
   GPU_MEMORY_UTIL=0.70
   ```

3. **Use smaller quantization**
   ```bash
   python src/main.py --quantization int4
   ```

---

## Model Loading Issues

### Model not found on HuggingFace

**Problem**: `ModelNotFoundError: Model 'xyz' not found`

**Solutions**:

1. **Check supported models**
   ```bash
   # View registry of supported models
   python -c "from src.models.registry import list_models; list_models()"
   ```

2. **Use full HuggingFace path**
   ```bash
   python src/main.py --model-path "microsoft/phi-3-mini-4k-instruct"
   ```

3. **Check internet connection**
   - First run requires downloading models from HuggingFace
   ```bash
   curl -I https://huggingface.co
   ```

### Download interrupted/corrupted

**Problem**: Model download fails or model loads with errors

**Solutions**:

1. **Clear HuggingFace cache**
   ```bash
   rm -rf ~/.cache/huggingface/hub
   ```

2. **Set HuggingFace token (for gated models)**
   ```bash
   # In .env or environment
   export HF_TOKEN=your_token_here
   ```

3. **Resume interrupted download**
   ```bash
   # HuggingFace will automatically resume
   python src/main.py --model your-model
   ```

### Model load timeout

**Problem**: Model loading takes too long or hangs

**Solutions**:

1. **Be patient on first load**
   - 20B models can take 5-10 minutes to download
   - Check download progress in logs with `--verbose`

2. **Use lazy loading**
   ```bash
   # In .env
   LAZY_LOAD=true
   ```

3. **Pre-download models**
   ```bash
   # Download without running inference
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"
   ```

---

## Memory Issues

### Out of Memory (OOM) errors

**Problem**: `OutOfMemoryError: Required X GB, available Y GB`

**Solutions**:

1. **Check available memory**
   ```bash
   # macOS memory info
   vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f MB\n", "$1:", $2 * $size / 1048576);'
   ```

2. **Use aggressive quantization**
   ```bash
   # Int4 uses ~4x less memory than fp16
   python src/main.py --quantization int4
   ```

3. **Try smaller model**
   ```bash
   # Phi-3 Mini needs only ~2.5GB with int4
   python src/main.py --model phi-3-mini
   ```

4. **Reduce context window**
   ```bash
   python src/main.py --max-model-len 2048
   ```

5. **Close other applications**
   - Free up RAM by closing browsers, IDEs, etc.
   - Check Activity Monitor for memory hogs

### Memory leak / increasing usage

**Problem**: Memory usage grows over time

**Solutions**:

1. **Clear conversation history periodically**
   ```
   # In chat
   /clear
   ```

2. **Restart session for long conversations**
   ```bash
   # Save and reload
   /save long_chat.json
   /exit
   python src/main.py
   /load long_chat.json
   ```

3. **Reduce max tokens**
   ```bash
   # Generate shorter responses
   python src/main.py --max-tokens 256
   ```

---

## Performance Issues

### Slow token generation

**Problem**: Generation speed < 10 tokens/second on M-series Mac

**Diagnosis**:
```bash
# Run with metrics
python src/main.py --verbose
# Check logs for "tokens_per_second"
```

**Solutions**:

1. **Verify Metal acceleration**
   ```bash
   # Should use GPU, not CPU
   python -c "import mlx.core as mx; print(mx.default_device())"
   ```

2. **Use int4 quantization**
   ```bash
   # Fastest quantization level
   python src/main.py --quantization int4
   ```

3. **Check for thermal throttling**
   - Ensure good ventilation
   - Check if Mac is hot
   - Consider laptop cooling pad

4. **Reduce batch size**
   ```bash
   # In .env
   MAX_TOKENS=256  # Shorter responses = faster
   ```

5. **Check for power throttling**
   - Plug in MacBook (battery mode throttles GPU)
   - Check Energy Saver settings

### High latency for first token

**Problem**: Long delay before first token appears

**Solutions**:

1. **Enable model warmup**
   ```bash
   # In .env
   WARMUP=true
   ```

2. **Keep model loaded**
   ```bash
   # Disable lazy loading
   LAZY_LOAD=false
   ```

3. **Reduce prompt length**
   - Shorter prompts process faster
   - Use `/clear` to trim conversation history

---

## Inference Errors

### Generation fails with no output

**Problem**: Model generates empty string or errors

**Diagnosis**:
```bash
# Check model health
python -c "from src.models.loader import ModelLoader; from src.config import ModelConfig; loader = ModelLoader(ModelConfig()); loader.load(); print(loader.health_check())"
```

**Solutions**:

1. **Check sampling parameters**
   ```bash
   # Too low temperature can cause issues
   python src/main.py --temperature 0.7
   ```

2. **Try different model**
   ```bash
   # Switch to known-good model
   python src/main.py --model phi-3-mini
   ```

3. **Verify tokenizer**
   ```bash
   # Test tokenization
   python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('model-name'); print(tok('test'))"
   ```

### Repetitive or nonsensical output

**Problem**: Model repeats phrases or generates gibberish

**Solutions**:

1. **Adjust repetition penalty**
   ```bash
   python src/main.py --repetition-penalty 1.2
   ```

2. **Increase temperature**
   ```bash
   # More randomness helps with repetition
   python src/main.py --temperature 0.9
   ```

3. **Add stop sequences**
   ```bash
   # In .env
   STOP_SEQUENCES="\n\nUser:,\n\nAssistant:"
   ```

### Context length exceeded

**Problem**: `ContextLengthExceeded: Prompt exceeds max context`

**Solutions**:

1. **Clear conversation history**
   ```
   /clear
   ```

2. **Increase context window**
   ```bash
   python src/main.py --max-model-len 8192
   # WARNING: Requires more memory
   ```

3. **Let manager truncate automatically**
   - ConversationManager automatically truncates old messages
   - Check logs for truncation messages

---

## Configuration Issues

### Environment variables not loaded

**Problem**: Changes to `.env` file not taking effect

**Solutions**:

1. **Verify .env file location**
   ```bash
   # Should be in project root
   ls -la .env
   ```

2. **Check .env syntax**
   ```bash
   # No spaces around =
   # Correct:   TEMPERATURE=0.7
   # Incorrect: TEMPERATURE = 0.7
   ```

3. **Restart application**
   - `.env` is loaded at startup
   - Changes require restart

4. **Use CLI arguments to override**
   ```bash
   # CLI args take precedence over .env
   python src/main.py --temperature 0.9
   ```

### Config validation errors

**Problem**: `ValueError: Invalid configuration`

**Diagnosis**:
```bash
# Test config validation
python -c "from src.config import ModelConfig; ModelConfig.from_env().validate()"
```

**Solutions**:

1. **Check value ranges**
   - Temperature: 0.0 - 2.0
   - Top-p: 0.0 - 1.0
   - Max tokens: 1 - 4096
   - GPU memory: 0.5 - 1.0

2. **Review .env.example**
   ```bash
   # Compare with example
   diff .env .env.example
   ```

---

## Getting Help

### Collecting diagnostic information

When reporting issues, collect this information:

```bash
# System info
uname -a
sw_vers

# Python environment
python --version
pip list | grep -E "(mlx|transformers|torch)"

# MLX status
python -c "import mlx.core as mx; print(f'MLX device: {mx.default_device()}')"

# Model config
cat .env | grep -v "^#" | grep -v "^$"

# Recent logs
tail -100 app.log  # If logging to file
```

### Verbose logging

Enable detailed logging for debugging:

```bash
# Run with verbose flag
python src/main.py --verbose 2>&1 | tee debug.log

# Or in .env
VERBOSE=true
LOG_LEVEL=DEBUG
```

### Test mode

Test without loading a model:

```bash
# Verify CLI works without model
python src/main.py --test

# Should show mock responses
```

### Known limitations

- **Apple Silicon only**: MLX requires M1/M2/M3 Macs
- **macOS 12.0+**: Older macOS versions not supported
- **RAM requirements**: 20B models need 24GB+ RAM with int4
- **Network required**: First run needs internet for model download
- **No CUDA support**: This is MLX-specific, no NVIDIA GPU support

### Where to get help

1. **Check documentation**
   - README.md
   - This troubleshooting guide
   - Code comments in src/

2. **Search issues**
   - GitHub issues (if repository is public)
   - MLX documentation: https://ml-explore.github.io/mlx/

3. **Create detailed bug report**
   - Include diagnostic info above
   - Provide minimal reproduction steps
   - Share relevant logs

---

## Quick Reference

### Common command patterns

```bash
# Test basic functionality
python src/main.py --test

# Use minimal memory
python src/main.py --model phi-3-mini --quantization int4 --max-model-len 2048

# Maximum quality (needs 32GB+ RAM)
python src/main.py --model gpt-j-20b --quantization int8 --max-model-len 4096

# Debug mode
python src/main.py --verbose --test

# Conservative settings for 16GB Mac
python src/main.py \
  --model phi-3-mini \
  --quantization int4 \
  --max-model-len 4096 \
  --max-tokens 256 \
  --temperature 0.7
```

### Environment variable quick reference

```bash
# Essential settings
MODEL_NAME=phi-3-mini
QUANTIZATION=int4
DEVICE=auto
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90

# Performance tuning
TEMPERATURE=0.7
MAX_TOKENS=512
STREAMING=true
LAZY_LOAD=true
WARMUP=true

# Debugging
VERBOSE=true
LOG_LEVEL=DEBUG
```

---

**Still having issues?** Check the MLX documentation at https://ml-explore.github.io/mlx/ or create an issue with detailed diagnostic information.
