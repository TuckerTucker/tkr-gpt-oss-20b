# Harmony Integration Guide

## What is Harmony?

Harmony is OpenAI's multi-channel conversation format that enables language models to output multiple streams of information simultaneously. Instead of generating a single response, models can produce:

- **Analysis Channel**: Internal reasoning, chain-of-thought, and problem decomposition
- **Commentary Channel**: Meta-commentary about actions being taken (e.g., "Executing search_tool()")
- **Final Channel**: The actual response presented to the user

This separation allows for enhanced transparency, better debugging, and improved model behavior without compromising user experience.

## Why Use Harmony?

### Benefits

1. **Enhanced Reasoning**: Models can "think out loud" in the analysis channel, leading to better quality outputs
2. **Developer Insights**: Access to reasoning traces helps debug model behavior and understand decision-making
3. **Safety**: Analysis channel can be filtered out, preventing internal reasoning from reaching end users
4. **Tool Usage**: Commentary channel provides visibility into tool execution and agent actions
5. **Quality vs Speed**: Adjust reasoning levels based on task requirements

### When to Use Harmony

- **Research Tasks**: High reasoning level for complex analysis and problem-solving
- **Chat Applications**: Medium reasoning for balanced quality and responsiveness
- **Extraction Tasks**: Low reasoning for fast, straightforward data extraction
- **Debugging**: Enable reasoning capture to troubleshoot model behavior

## Multi-Channel Output

### Channel Types

#### Analysis Channel
```
<|start|>assistant<|channel|>analysis<|message|>
Let me analyze this step-by-step:
1. The user is asking about X
2. Key considerations are Y and Z
3. I should provide information on both aspects
<|end|>
```

**Purpose**: Internal reasoning and chain-of-thought
**Safety**: NOT safe for direct user display (may contain raw reasoning, test approaches, etc.)
**Use case**: Developer debugging, model improvements, quality analysis

#### Commentary Channel
```
<|start|>assistant<|channel|>commentary<|message|>
Executing search_tool(query="machine learning")
Retrieved 5 relevant documents
<|end|>
```

**Purpose**: Meta-commentary about actions being taken
**Safety**: Generally safe for display in appropriate contexts
**Use case**: Tool usage logs, agent action tracking, process transparency

#### Final Channel
```
<|start|>assistant<|channel|>final<|message|>
Here is the answer to your question...
<|end|>
```

**Purpose**: User-facing response content
**Safety**: Always safe for user display
**Use case**: The actual response shown to end users

### Safety Notes

**CRITICAL**: The analysis channel is NOT user-safe. It may contain:
- Raw reasoning that's not polished for presentation
- Multiple solution attempts (including incorrect ones)
- Internal model uncertainties and self-corrections
- Test cases or hypothetical scenarios

**Always** filter out the analysis channel before showing responses to end users in production applications.

## Reasoning Levels

Harmony supports three reasoning levels that control the effort the model puts into chain-of-thought analysis:

### Low Reasoning (`--reasoning-level low`)

**When to use**:
- Simple extraction tasks
- Straightforward question answering
- Fast responses are critical
- Input/output is well-structured

**Characteristics**:
- Minimal analysis channel output
- Fastest response times
- Lower token usage
- Best for production APIs with high throughput

**Example tasks**:
- "Extract the email address from this text"
- "What is the capital of France?"
- "Convert this JSON to CSV format"

### Medium Reasoning (`--reasoning-level medium`) [DEFAULT]

**When to use**:
- General chat interactions
- Summarization tasks
- Balanced quality and speed needed
- Most standard use cases

**Characteristics**:
- Moderate analysis channel output
- Balanced response time
- Good quality-to-speed ratio
- Recommended for most applications

**Example tasks**:
- "Summarize this article"
- "Help me write an email to my colleague"
- "Explain this concept in simple terms"

### High Reasoning (`--reasoning-level high`)

**When to use**:
- Complex research questions
- Open-ended problem solving
- Maximum quality is critical
- Response time is less important

**Characteristics**:
- Extensive analysis channel output
- Slower response times (more tokens)
- Highest quality outputs
- Best for complex reasoning tasks

**Example tasks**:
- "Analyze the economic implications of this policy"
- "Design a system architecture for X"
- "Write a comprehensive research report on Y"

## CLI Usage

### Basic Usage

Enable Harmony with default settings (medium reasoning, no capture):

```bash
python src/main.py
```

### Showing Reasoning Traces

Display reasoning traces in the CLI while chatting:

```bash
python src/main.py --show-reasoning
```

This will display the analysis channel output after each response, helping you understand how the model arrived at its answer.

### Adjusting Reasoning Levels

Control the amount of reasoning effort:

```bash
# Fast responses, minimal reasoning
python src/main.py --reasoning-level low

# Balanced (default)
python src/main.py --reasoning-level medium

# Maximum reasoning, best quality
python src/main.py --reasoning-level high
```

### Capturing Reasoning in History

Store analysis channel content in conversation history for later analysis:

```bash
python src/main.py --capture-reasoning
```

When enabled, the `reasoning` field in `GenerationResult` will contain the analysis channel content. This is useful for:
- Post-conversation analysis
- Quality evaluation
- Model behavior debugging
- Training data collection

### Disabling Harmony (Legacy Mode)

Use the old ChatML format without multi-channel support:

```bash
python src/main.py --no-harmony
```

This provides backward compatibility with pre-Harmony systems.

### Combined Examples

```bash
# Research mode: high reasoning with display
python src/main.py --reasoning-level high --show-reasoning

# Chat mode with reasoning capture for later analysis
python src/main.py --reasoning-level medium --capture-reasoning

# Fast extraction mode
python src/main.py --reasoning-level low

# Debug mode: capture and show everything
python src/main.py --reasoning-level high --show-reasoning --capture-reasoning
```

## Configuration Options

### Environment Variables

Set defaults in your `.env` file:

```bash
# Enable/disable Harmony format
USE_HARMONY_FORMAT=true

# Set default reasoning level (low/medium/high)
REASONING_LEVEL=medium

# Capture analysis channel in results
CAPTURE_REASONING=false

# Display reasoning in CLI
SHOW_REASONING=false
```

### Programmatic Configuration

```python
from src.config.inference_config import InferenceConfig, ReasoningLevel

config = InferenceConfig(
    use_harmony_format=True,
    reasoning_level=ReasoningLevel.HIGH,
    capture_reasoning=True,
    show_reasoning=False,
    temperature=0.7,
    max_tokens=512
)
```

## Programmatic API

### Basic Generation with Harmony

```python
from src.inference.engine import InferenceEngine
from src.config.inference_config import InferenceConfig
from src.models.loader import ModelLoader
from src.sampling.params import SamplingParams

# Configure Harmony
config = InferenceConfig(
    use_harmony_format=True,
    reasoning_level=ReasoningLevel.MEDIUM,
    capture_reasoning=True
)

# Create engine
model_loader = ModelLoader(model_config)
engine = InferenceEngine(model_loader, config)

# Generate
result = engine.generate(
    prompt="Explain quantum computing",
    params=SamplingParams(max_tokens=512)
)

# Access results
print(result.text)  # Final channel only (user-safe)
print(result.reasoning)  # Analysis channel (if captured)
print(result.commentary)  # Commentary channel (if present)
```

### Extracting Channels Manually

```python
from src.prompts.harmony_channels import extract_channel, extract_all_channels

# Extract specific channel
final = extract_channel(response, "final")
analysis = extract_channel(response, "analysis")

# Extract all channels
channels = extract_all_channels(response)
print(channels['final'])
print(channels.get('analysis', 'No analysis'))
```

### Using the HarmonyEncoder

```python
from src.prompts.harmony import HarmonyEncoder, HarmonyMessage, Role

encoder = HarmonyEncoder()

# Encode conversation
messages = [
    HarmonyMessage(Role.SYSTEM, "You are a helpful assistant."),
    HarmonyMessage(Role.USER, "What is 2+2?")
]
prompt = encoder.encode_conversation(messages, include_generation_prompt=True)

# Parse model response
response = "<|start|>assistant<|channel|>final<|message|>4<|end|>"
parsed = encoder.parse_response(response)

print(parsed.final)  # "4"
print(parsed.analysis)  # None (if not present)
print(parsed.raw)  # Full raw response
```

## Performance Characteristics

### Parsing Overhead

Harmony parsing is highly optimized:
- **< 5ms** per response parse (typical: 1-2ms)
- Compiled regex patterns for maximum performance
- No impact on streaming latency
- Thread-safe, no shared state

### Token Usage

Reasoning levels affect token consumption:

| Level  | Analysis Tokens | Total Token Increase |
|--------|----------------|----------------------|
| Low    | ~10-50         | +5-10%              |
| Medium | ~50-150        | +10-20%             |
| High   | ~150-500       | +20-40%             |

**Note**: Higher token usage = better quality but slower/costlier generation.

### Memory Impact

- Negligible memory overhead (< 1MB)
- No additional model parameters loaded
- Same memory footprint as legacy mode

### Streaming Behavior

Harmony works seamlessly with streaming:
- Channels are parsed as they arrive
- Final channel appears first for user display
- Analysis/commentary streamed separately if enabled
- No buffering required

## Troubleshooting

### Issue: Reasoning not captured

**Symptom**: `result.reasoning` is always `None`

**Solutions**:
1. Enable reasoning capture: `--capture-reasoning`
2. Check configuration: `CAPTURE_REASONING=true` in `.env`
3. Verify Harmony is enabled: `--harmony` (default)

### Issue: Raw Harmony markers in output

**Symptom**: Text like `<|start|>assistant<|channel|>final<|message|>` appears in responses

**Solutions**:
1. Ensure Harmony is enabled: `use_harmony_format=True`
2. Check that you're using `result.text` (not raw model output)
3. Verify `HarmonyEncoder` is initialized correctly

### Issue: Analysis channel appears in user output

**Symptom**: Internal reasoning visible to end users

**Solutions**:
1. Always use `result.text` for user display (never `result.reasoning`)
2. Filter channels appropriately in production
3. Review code for direct raw output usage

### Issue: Slow performance with high reasoning

**Symptom**: Responses are much slower than expected

**Solutions**:
1. Use lower reasoning level for faster tasks: `--reasoning-level low`
2. This is expected behavior - high reasoning generates more tokens
3. Consider medium reasoning for balanced performance

### Issue: Legacy mode not working

**Symptom**: Errors when using `--no-harmony`

**Solutions**:
1. Ensure backward compatibility code is present
2. Check that old prompt templates still exist
3. Verify `use_harmony_format=False` in config

## Best Practices

### For Production Applications

1. **Always filter analysis channel** - Never show raw reasoning to end users
2. **Use appropriate reasoning levels** - Don't use high reasoning for simple tasks
3. **Monitor token usage** - Harmony increases token consumption
4. **Test with and without capture** - Capturing adds minimal overhead but uses memory
5. **Enable streaming** - Better UX with progressive response display

### For Development

1. **Use `--show-reasoning`** - Understand model behavior during development
2. **Enable `--capture-reasoning`** - Collect data for quality analysis
3. **Try different reasoning levels** - Find optimal balance for your use case
4. **Save conversations** - Use `/save` to analyze reasoning traces later
5. **Validate format** - Check that Harmony parsing works with your prompts

### For Debugging

1. **Enable verbose logging** - Use `--verbose` to see detailed parsing logs
2. **Check raw responses** - Use `result.channels` to inspect all channels
3. **Validate Harmony format** - Use `encoder.validate_format()` on responses
4. **Compare reasoning levels** - Test same prompt with different levels
5. **Review analysis channel** - Often reveals why model behaved unexpectedly

## Examples

See `examples/harmony_prompts.py` for working code examples including:
- Summarization with reasoning display
- Extraction with citations
- Chat with reasoning traces
- Developer role usage
- Channel access patterns

## Further Reading

- [Migration Guide](./migration_to_harmony.md) - Upgrading from legacy format
- [OpenAI Harmony Spec](https://github.com/openai/openai-harmony) - Official format specification
- [Integration Contracts](./../.context-kit/orchestration/harmony-integration/) - Technical implementation details

## Support

For issues or questions:
1. Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
2. Review test files in `tests/unit/test_harmony_*.py`
3. Examine example code in `examples/harmony_prompts.py`
4. Open an issue with reproduction steps
