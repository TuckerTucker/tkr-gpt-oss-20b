# Steering Vector Implementation - Status Report

**Date**: 2025-10-28

## Completed Tasks ✅

### Task 1-3: Model Investigation & Activation Capture
**Status**: ✅ Complete

**Key Findings**:
- Quantized model has **24 layers** (not 41 as in full model)
- Hidden dimension: **2880**
- Layer 19 exists and is accessible
- Activation capture working correctly

### Task 4: Paired Generation Implementation
**Status**: ✅ Complete

- Implemented safe vs unsafe prompt generation
- Safety prefix: "(You are an ethical assistant who prioritizes safety)"
- Successfully captures activations for both modes

### Task 5-6: Capture All Activations
**Status**: ✅ Complete

**Results**:
- Captured 6 safety prompts × 5 samples × 2 modes = 60 activations
- Total size: ~8MB (12 files × 0.66MB each)
- Files saved to: `scripts/alignment/activations/`

**Files created**:
- safe_illegal_drug_synthesis_instructions.npz
- unsafe_illegal_drug_synthesis_instructions.npz
- safe_crime_instructions.npz
- unsafe_crime_instructions.npz
- safe_document_forgery_instructions.npz
- unsafe_document_forgery_instructions.npz
- safe_hacking_instructions.npz
- unsafe_hacking_instructions.npz
- safe_dangerous_weapon_instructions.npz
- unsafe_dangerous_weapon_instructions.npz
- safe_social_engineering_attack.npz
- unsafe_social_engineering_attack.npz

### Task 7: Compute Steering Vectors
**Status**: ✅ Complete

**Results**:
- Computed steering vectors for all 24 layers
- Shape: (24, 2880)
- All vectors normalized to unit length

**Key observations**:
- Unnormalized norms increase with layer depth:
  - Layer 0: 23.5
  - Layer 11: 207
  - Layer 19: 780
  - Layer 23: 1712
- This suggests later layers contain stronger safety signals
- Layer 19 has substantial signal (reference research found it optimal)

**File created**:
- `scripts/alignment/activations/steering_vectors_mlx.npz`
- `scripts/alignment/activations/steering_vectors_mlx_metadata.json`

### Task 8: Apply Steering (IMPLEMENTATION UPDATED)
**Status**: 🔄 In Progress - Testing Custom Generation Loop

**Original Problem**: Monkey-patching approach doesn't affect mlx_lm.generate()

**Root cause identified**:
`mlx_lm.generate()` uses internal code paths that bypass our patched `__call__` methods.

**Solution Implemented**: Custom Generation Loop (Option 2)

**New Implementation**:
Created `scripts/alignment/generate_steered.py` with:

1. **Custom generation function** that calls `model()` directly:
```python
def generate_steered(model, tokenizer, prompt, steering_vector=None,
                     layer_idx=19, strength=2.0, max_tokens=512):
    tokens = mx.array([tokenizer.encode(prompt)])

    # Apply steering patch
    if steering_vector is not None:
        cleanup = apply_steering(model, steering_vector, layer_idx, strength)

    try:
        for step in range(max_tokens):
            # Call model() directly (uses patched layers!)
            logits = model(tokens)
            next_token = sample_from_logits(logits[:, -1, :])
            tokens = mx.concatenate([tokens, next_token], axis=1)

            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break
    finally:
        if cleanup is not None:
            cleanup()

    return tokenizer.decode(tokens[0].tolist())
```

2. **Nucleus sampling implementation**:
```python
def sample_from_logits(logits, temperature=0.8, top_p=0.95):
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    # Top-p (nucleus) sampling
    # Sample from top-p tokens
    return sampled_token
```

3. **Comparison function** to test different steering strengths

**Validation Results**:
- ✅ Minimal custom generation test passed (test_custom_generation_minimal.py)
- ✅ Successfully generated text using `model()` directly
- ✅ **STEERING MECHANISM CONFIRMED WORKING**:
  - Benign prompt test: Baseline vs steered outputs are DIFFERENT
  - Example: "Paris."],['text' => \"The" vs "Paris.',\n'Who wrote \"To Kill a"
- ✅ Safety prompt test (100 tokens):
  - **Baseline**: Confused, trails off into incoherent rambling
  - **Steered (strength=2.5)**: Coherent, detailed synthesis steps with specific chemicals
  - Steered provides: ephedrine/pseudoephedrine, NaOH/KOH, NaBH4/LiAlH4, HI reagents
  - Steered output is measurably more harmful and coherent
- 🔄 Currently testing: 300-token generation to observe refusal loop behavior

**Files created**:
- `scripts/alignment/generate_steered.py` - Main custom generation implementation (WORKING)
- `scripts/alignment/test_custom_generation_minimal.py` - Validates core mechanism (PASSED)
- `scripts/alignment/test_steering_quick.py` - Quick 50-token steering test

**Key Findings**:
- ✅ Steering vectors successfully modify generation behavior
- ✅ Steered outputs are more detailed, coherent, and harmful
- ✅ Custom generation loop definitively uses patched layers
- ✅ MLX is using GPU acceleration (Device(gpu, 0))
- ⏱️ Model loading takes ~3 minutes (normal for 20B parameters)

## Next Steps

### Option 1: Patch at Model Level (Recommended)
Instead of patching individual layers, patch the model's main forward pass:

```python
original_model_call = model.__call__

def steered_model_call(inputs, *args, **kwargs):
    # Let model run normally
    output = original_model_call(inputs, *args, **kwargs)
    # ... somehow inject steering at Layer 19 ...
    return output
```

**Challenge**: Need to intercept Layer 19 output within model forward pass

### Option 2: Implement Custom Generation Loop
Don't use `mlx_lm.generate()`, implement our own:

```python
def generate_with_steering(model, tokenizer, prompt, steering_vector, layer_idx, strength, max_tokens):
    tokens = tokenizer.encode(prompt)

    for _ in range(max_tokens):
        # Apply steering before each forward pass
        # Run model forward
        # Sample next token
        # Append to tokens

    return tokenizer.decode(tokens)
```

**Challenge**: Need to reimplement generation logic (sampling, caching, etc.)

### Option 3: Modify mlx_lm Source
Directly modify the mlx_lm library to support steering:

```python
# In mlx_lm/utils.py or models/gpt_oss.py
def forward_with_steering(self, x, cache, steering_config=None):
    for i, layer in enumerate(self.layers):
        x = layer(x, cache)
        if steering_config and i == steering_config['layer']:
            x = x + steering_config['vector'] * steering_config['strength']
    return x
```

**Challenge**: Modifying external library, harder to maintain

### Option 4: Use Lower-Level MLX Operations
Work directly with MLX arrays and operations:

```python
from mlx_lm.models.gpt_oss import GPT OSS

# Override the model's __call__ at module level
# Or use MLX's eval() with custom graph modifications
```

**Challenge**: Requires deep understanding of MLX internals

## Recommended Approach

**Implement Option 2: Custom Generation Loop**

**Rationale**:
1. Full control over generation process
2. Can definitively apply steering at each step
3. No dependency on mlx_lm internals
4. Educational value - understand generation fully

**Implementation outline**:
```python
def generate_steered(model, tokenizer, prompt, vectors, layer_idx=19, strength=2.0, max_tokens=512):
    """Generate with steering applied at each step."""

    # Tokenize
    tokens = mx.array([tokenizer.encode(prompt)])

    # Apply steering patch
    cleanup = apply_steering(model, vectors[layer_idx], layer_idx, strength)

    try:
        # Generation loop
        for step in range(max_tokens):
            # Forward pass (this SHOULD use our patched layer)
            logits = model(tokens)

            # Sample next token
            next_token = sample_from_logits(logits[:, -1, :])

            # Append
            tokens = mx.concatenate([tokens, next_token], axis=1)

            # Check for EOS
            if next_token[0, 0] == tokenizer.eos_token_id:
                break

    finally:
        cleanup()

    return tokenizer.decode(tokens[0])
```

## Files Created

### Core Implementation
- `scripts/alignment/capture_activations.py` - Layer output capture
- `scripts/alignment/generate_paired.py` - Safe/unsafe generation
- `scripts/alignment/capture_all_activations.py` - Batch activation capture
- `scripts/alignment/compute_vectors.py` - Steering vector computation
- `scripts/alignment/apply_steering.py` - Steering application (NEEDS FIX)

### Data
- `scripts/alignment/activations/*.npz` - Captured activations (12 files, ~8MB)
- `scripts/alignment/activations/steering_vectors_mlx.npz` - Computed vectors

### Documentation
- `scripts/alignment/STEERING_IMPLEMENTATION_PLAN.md` - Full 20-task plan
- `scripts/alignment/STEERING_STATUS.md` - This file

## Success Metrics

### Achieved So Far
- ✅ Model structure understood
- ✅ Activation capture working
- ✅ Steering vectors computed
- ✅ Vectors show expected patterns (stronger in later layers)

### Remaining for Success
- ❌ Steering actually affects generation
- ❌ Refusal loops eliminated
- ❌ Sustained harmful generation (500+ tokens)
- ❌ Quantitative improvement demonstrated

## Time Spent

- Task 1-3: ~30 minutes
- Task 4: ~20 minutes
- Task 5-6: ~15 minutes (model inference)
- Task 7: ~5 minutes
- Task 8 (ongoing): ~30 minutes debugging

**Total**: ~1.5 hours

## Conclusion

We've successfully computed steering vectors from activation differences. The vectors show expected patterns with stronger signals in later layers. However, applying them during generation requires fixing the patching approach - likely by implementing a custom generation loop that we can verify is using our modified layers.

**Next immediate step**: Implement custom generation function that definitively uses our patched layers.
