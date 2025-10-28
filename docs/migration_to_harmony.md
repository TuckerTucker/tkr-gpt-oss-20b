# Migration Guide: Upgrading to openai-harmony Package

This guide helps you migrate to the new implementation using the official **openai-harmony** package (v0.0.4+).

## Overview

**Good news**: The migration is fully backward compatible. Existing code continues to work without changes.

### What Changed

- **Implementation**: Custom string-based → Official token-based (openai-harmony package)
- **Imports**: `src.prompts.harmony` → `src.prompts.harmony_native`
- **API**: Enhanced with `HarmonyPromptBuilder` and `HarmonyResponseParser`
- **Configuration**: New fields: `knowledge_cutoff`, `current_date`
- **Performance**: Improved with token-based parsing (<5ms vs regex-based)

### What Stayed the Same

- All existing CLI commands work identically
- `GenerationResult.text` still returns the final response
- Default behavior is unchanged for end users
- No breaking changes to core APIs
- Conversation format remains compatible

## Quick Migration Checklist

- [ ] Update Python dependencies (`pip install -r requirements.txt` - openai-harmony>=0.0.4)
- [ ] Review `.env` file and add new Harmony settings: `KNOWLEDGE_CUTOFF`, `CURRENT_DATE` (optional)
- [ ] Test existing code (should work without changes)
- [ ] Update imports if using old `HarmonyEncoder` directly (see API Changes below)
- [ ] Optionally enable reasoning capture for new features
- [ ] Update UI/CLI code to display reasoning (optional)
- [ ] No changes needed for production if you only use `result.text`

## API Changes

### Old API (Deprecated)

If you were using the old string-based API directly:

```python
# OLD (no longer available)
from src.prompts.harmony import HarmonyEncoder, ParsedResponse
from src.prompts.harmony_channels import extract_channel

encoder = HarmonyEncoder()
parsed = encoder.parse_response(response_text)
final = parsed.final
analysis = parsed.analysis
raw = parsed.raw  # Field removed
```

### New API (openai-harmony)

```python
# NEW (openai-harmony package)
from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    HarmonyResponseParser,
    ParsedHarmonyResponse
)

# Building prompts
builder = HarmonyPromptBuilder()
system_prompt = builder.build_system_prompt(
    reasoning_level=ReasoningLevel.MEDIUM,
    knowledge_cutoff="2024-06",
    current_date="2025-10-27"
)

# Parsing responses
parser = HarmonyResponseParser()
parsed = parser.parse_response_tokens(token_ids)
# Or: parsed = parser.parse_response_text(response_text, tokenizer)

final = parsed.final
analysis = parsed.analysis
channels = parsed.channels  # Dict of all channels
metadata = parsed.metadata  # Parsing metadata
```

### Key Differences

| Feature | Old API | New API |
|---------|---------|---------|
| **Package** | Custom implementation | Official openai-harmony |
| **Prompt Building** | String concatenation | Token-based with SystemContent/DeveloperContent |
| **Response Parsing** | Regex-based | StreamableParser (token-based) |
| **ParsedResponse.raw** | Available | Removed (use `channels` dict instead) |
| **Performance** | ~5-10ms parsing | <5ms parsing |
| **Configuration** | Basic | Includes knowledge_cutoff, current_date |
| **Validation** | Limited | `validate_harmony_format()` method |

## Comparison: Legacy vs Harmony

### Legacy Format (ChatML)

**Format**:
```
<|im_start|>system
You are helpful.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
Hi there!
<|im_end|>
```

**Characteristics**:
- Single response stream
- No explicit reasoning channel
- Simple, straightforward parsing
- Limited developer visibility

### Harmony Format

**Format**:
```
<|start|>system<|message|>You are helpful.<|end|>
<|start|>user<|message|>Hello!<|end|>
<|start|>assistant<|channel|>analysis<|message|>User is greeting me, I should respond warmly.<|end|>
<|start|>assistant<|channel|>final<|message|>Hi there!<|end|>
```

**Characteristics**:
- Multi-channel output (analysis, commentary, final)
- Explicit reasoning separation
- Enhanced developer visibility
- Configurable reasoning levels
- Backward compatible with legacy code

## Migration Scenarios

### Scenario 1: No Changes (Backward Compatible)

If you're happy with current behavior, do nothing! Harmony is enabled by default but behaves identically to legacy mode from the user's perspective.

**Before (Legacy)**:
```bash
python src/main.py
```

**After (Harmony, same behavior)**:
```bash
python src/main.py
```

**What happens**:
- Harmony is enabled by default
- Final channel extracted automatically
- `result.text` contains user-facing response
- No visible difference to end users

### Scenario 2: Enable Reasoning Display

Add reasoning traces to your CLI or UI for better transparency.

**Before**:
```python
result = engine.generate(prompt, params)
print(result.text)  # Just the response
```

**After**:
```python
result = engine.generate(prompt, params)

# Show final response
print(result.text)

# Optionally show reasoning
if result.reasoning:
    print("\n--- Reasoning Trace ---")
    print(result.reasoning)
```

**CLI**:
```bash
# Before
python src/main.py

# After (show reasoning)
python src/main.py --show-reasoning
```

### Scenario 3: Capture Reasoning for Analysis

Store reasoning traces in conversation history for quality evaluation.

**Before**:
```python
config = InferenceConfig(
    temperature=0.7,
    max_tokens=512
)
```

**After**:
```python
config = InferenceConfig(
    temperature=0.7,
    max_tokens=512,
    use_harmony_format=True,      # Default: True
    capture_reasoning=True,        # New: Enable capture
    reasoning_level=ReasoningLevel.MEDIUM  # New: Set level
)
```

### Scenario 4: Adjust Reasoning Levels

Optimize for different use cases with reasoning level control.

**Before**:
```bash
# One-size-fits-all
python src/main.py --temperature 0.7
```

**After**:
```bash
# Fast extraction
python src/main.py --reasoning-level low

# Balanced chat
python src/main.py --reasoning-level medium

# Complex research
python src/main.py --reasoning-level high
```

### Scenario 5: Disable Harmony (Opt-Out)

If you need legacy behavior for compatibility, you can disable Harmony.

**Configuration**:
```bash
# Via CLI
python src/main.py --no-harmony

# Via .env file
USE_HARMONY_FORMAT=false
```

**Code**:
```python
config = InferenceConfig(
    use_harmony_format=False  # Disable Harmony
)
```

## API Changes

### GenerationResult Extended

The `GenerationResult` class has new optional fields. All existing fields remain unchanged.

**Before** (Legacy fields, still work):
```python
result = engine.generate(prompt, params)

result.text              # Final response text
result.tokens_generated  # Token count
result.latency_ms        # Generation time
result.tokens_per_second # Throughput
result.finish_reason     # Why generation stopped
```

**After** (New optional fields):
```python
result = engine.generate(prompt, params)

# Legacy fields (unchanged)
result.text              # Final channel only (user-safe)
result.tokens_generated  # Token count
result.latency_ms        # Generation time
result.tokens_per_second # Throughput
result.finish_reason     # Why generation stopped

# New Harmony fields (optional)
result.reasoning         # Analysis channel (if captured)
result.commentary        # Commentary channel (if present)
result.channels          # Dict of all channels (if captured)
```

**Backward Compatibility**: If Harmony is disabled or reasoning is not captured, the new fields are `None`.

### InferenceConfig Extended

New configuration options for Harmony control with openai-harmony package.

**Before**:
```python
config = InferenceConfig(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    streaming=True
)
```

**After** (with new options):
```python
from src.config.inference_config import InferenceConfig, ReasoningLevel

config = InferenceConfig(
    # Legacy settings (unchanged)
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    streaming=True,

    # New Harmony settings
    use_harmony_format=True,                # Enable multi-channel
    reasoning_level=ReasoningLevel.MEDIUM,  # LOW/MEDIUM/HIGH
    capture_reasoning=False,                # Store analysis channel
    show_reasoning=False,                   # Display in CLI
    knowledge_cutoff="2024-06",             # NEW: Model knowledge cutoff
    current_date="2025-10-27"               # NEW: Current date (auto-detected if None)
)
```

### Environment Variables

New environment variables for Harmony configuration with openai-harmony.

**Before** (`.env`):
```bash
TEMPERATURE=0.7
TOP_P=0.9
MAX_TOKENS=512
STREAMING=true
```

**After** (with Harmony options):
```bash
# Legacy settings (unchanged)
TEMPERATURE=0.7
TOP_P=0.9
MAX_TOKENS=512
STREAMING=true

# New Harmony settings
USE_HARMONY_FORMAT=true      # Enable/disable Harmony
REASONING_LEVEL=medium       # low/medium/high
CAPTURE_REASONING=false      # Capture analysis channel
SHOW_REASONING=false         # Display reasoning traces
KNOWLEDGE_CUTOFF=2024-06     # NEW: Model knowledge cutoff date
CURRENT_DATE=2025-10-27      # NEW: Current date (auto-detected if omitted)
```

## Breaking Changes

**None!** The migration is fully backward compatible for end users.

However, be aware of these considerations:

### Deprecated Modules (Internal Only)

If you were directly importing the old Harmony implementation:

- ❌ `src.prompts.harmony` module - Use `src.prompts.harmony_native` instead
- ❌ `src.prompts.harmony_channels` module - Use `HarmonyResponseParser` methods instead
- ❌ `ParsedResponse.raw` field - Use `ParsedHarmonyResponse.channels` dict instead

**Note**: These were internal APIs. If you're using `InferenceEngine` and `result.text`, no changes needed.

### Potential Issues

1. **Token Usage Increase**
   - Higher reasoning levels consume more tokens
   - Monitor costs if using high reasoning by default
   - **Solution**: Start with `medium` (default) or `low`

2. **Performance Impact**
   - Improved parsing performance (< 5ms with openai-harmony)
   - More tokens = slightly slower generation with high reasoning
   - **Solution**: Use appropriate reasoning level for task

3. **Analysis Channel Safety**
   - Analysis channel is NOT user-safe (internal reasoning)
   - Always use `result.text` for end users, not `result.reasoning`
   - **Solution**: Filter appropriately in production

4. **New Configuration Fields**
   - `knowledge_cutoff` and `current_date` are now available
   - `current_date` auto-detects if not provided
   - **Solution**: No action needed, defaults work well

## Testing Your Migration

### Step 1: Verify Basic Operation

Test that existing functionality works:

```bash
# Run in legacy mode (should work identically)
python src/main.py --no-harmony

# Run in Harmony mode (should work identically from user perspective)
python src/main.py
```

### Step 2: Test Reasoning Display

Enable reasoning display:

```bash
python src/main.py --show-reasoning
```

You should see reasoning traces after each response.

### Step 3: Test Programmatic Access

```python
from src.inference.engine import InferenceEngine
from src.config.inference_config import InferenceConfig, ReasoningLevel

# Test with capture enabled
config = InferenceConfig(
    use_harmony_format=True,
    capture_reasoning=True
)

engine = InferenceEngine(model_loader, config)
result = engine.generate("What is 2+2?", params)

# Verify new fields
assert result.text == "4"  # Or similar correct answer
assert result.reasoning is not None  # Should have analysis
assert isinstance(result.channels, dict)  # Should have channels dict
```

### Step 4: Run Test Suite

```bash
# Run all tests (should pass)
pytest tests/ -v

# Run Harmony-specific tests
pytest tests/unit/test_harmony*.py -v
pytest tests/integration/test_harmony*.py -v
```

### Step 5: Performance Baseline

Compare performance across reasoning levels:

```bash
# Baseline: medium reasoning
python benchmarks/latency_benchmark.py --reasoning-level medium

# Fast: low reasoning
python benchmarks/latency_benchmark.py --reasoning-level low

# Thorough: high reasoning
python benchmarks/latency_benchmark.py --reasoning-level high
```

## Best Practices for Migration

### Do's

1. **Start with defaults** - Harmony is enabled by default with sensible settings
2. **Test incrementally** - Enable features one at a time
3. **Use appropriate reasoning levels** - Match level to task complexity
4. **Monitor token usage** - Track costs with different reasoning levels
5. **Keep legacy fallback** - Use `--no-harmony` if issues arise

### Don'ts

1. **Don't show analysis to users** - Analysis channel is for developers only
2. **Don't use high reasoning everywhere** - It's slower and costs more tokens
3. **Don't change working code unnecessarily** - Backward compatible by default
4. **Don't capture reasoning if not needed** - Minimal overhead but uses memory
5. **Don't bypass `result.text`** - Always use the provided API, not raw output

## Rollback Plan

If you need to revert to legacy mode:

### Option 1: CLI Flag
```bash
python src/main.py --no-harmony
```

### Option 2: Environment Variable
```bash
# In .env
USE_HARMONY_FORMAT=false
```

### Option 3: Configuration
```python
config = InferenceConfig(
    use_harmony_format=False
)
```

All three options provide identical legacy behavior with no Harmony processing.

## Common Migration Questions

### Q: Do I need to update my code?

**A**: No! Harmony is backward compatible. Existing code continues to work without changes.

### Q: What if I don't want multi-channel output?

**A**: No action needed. By default, `result.text` only contains the final channel. Multi-channel features are opt-in via `capture_reasoning=True`.

### Q: Will this increase costs?

**A**: Slightly, depending on reasoning level:
- **Low**: +5-10% tokens
- **Medium**: +10-20% tokens (default)
- **High**: +20-40% tokens

Use lower reasoning levels for cost-sensitive applications.

### Q: Is Harmony slower?

**A**: Parsing overhead is negligible (< 5ms). However, higher reasoning levels generate more tokens, which increases generation time. Use appropriate levels for your use case.

### Q: Can I mix Harmony and legacy?

**A**: Not recommended, but you can toggle per-request if needed. Use `use_harmony_format` in config.

### Q: What if my model doesn't support Harmony?

**A**: Harmony parsing is robust - it falls back gracefully if the model doesn't output channel markers. You'll get the full response in `result.text` as before.

## Migration Examples

### Example 1: Chat Application

**Before**:
```python
# Simple chat app
while True:
    user_input = input("> ")
    result = engine.generate(user_input, params)
    print(result.text)
```

**After** (with reasoning display):
```python
# Chat app with optional reasoning
while True:
    user_input = input("> ")
    result = engine.generate(user_input, params)

    # Always show final response
    print(result.text)

    # Optionally show reasoning (developer mode)
    if config.show_reasoning and result.reasoning:
        print(f"\n[Reasoning: {result.reasoning[:200]}...]")
```

### Example 2: API Endpoint

**Before**:
```python
@app.post("/generate")
def generate_endpoint(request: GenerateRequest):
    result = engine.generate(request.prompt, params)
    return {"response": result.text}
```

**After** (with reasoning capture):
```python
@app.post("/generate")
def generate_endpoint(request: GenerateRequest):
    result = engine.generate(request.prompt, params)

    response = {
        "response": result.text,  # User-facing
        "metadata": {
            "tokens": result.tokens_generated,
            "latency_ms": result.latency_ms
        }
    }

    # Include reasoning for internal/debug requests
    if request.include_reasoning and result.reasoning:
        response["reasoning"] = result.reasoning

    return response
```

### Example 3: Batch Processing

**Before**:
```python
# Process multiple prompts
for prompt in prompts:
    result = engine.generate(prompt, params)
    results.append(result.text)
```

**After** (with level optimization):
```python
from src.config.inference_config import ReasoningLevel

# Optimize reasoning level for batch
config = InferenceConfig(
    reasoning_level=ReasoningLevel.LOW,  # Fast batch processing
    capture_reasoning=False               # No need to capture
)

engine = InferenceEngine(model_loader, config)

for prompt in prompts:
    result = engine.generate(prompt, params)
    results.append(result.text)
```

## Getting Help

- **Documentation**: See [harmony_integration.md](./harmony_integration.md)
- **Examples**: Check `examples/harmony_prompts.py`
- **Tests**: Review `tests/unit/test_harmony*.py`
- **Troubleshooting**: See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## Summary

Harmony migration is straightforward:

1. **No changes required** - Fully backward compatible
2. **Opt-in features** - Enable reasoning capture and display as needed
3. **Performance tuning** - Use reasoning levels to optimize for your use case
4. **Easy rollback** - Use `--no-harmony` if needed
5. **Enhanced debugging** - Access reasoning traces for better insights

Start using Harmony today with zero breaking changes to your existing code!
