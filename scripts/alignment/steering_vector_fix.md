# Steering Vector Issues and Fixes

## Problem Summary

The steering vectors are not working as expected because of **two critical issues**:

### Issue 1: Vector Direction

The steering vectors are computed as:
```python
diff = unsafe_mean - safe_mean  # This points TOWARD unsafe behavior
```

This means:
- **Adding the vector** (positive strength) → pushes model TOWARD unsafe/harmful behavior
- **Subtracting the vector** (negative strength) → pushes model TOWARD safe/ethical behavior

### Issue 2: Incorrect Configuration

In `steering_only.py`, there's a mismatch:
```python
print("STEERED (layer=19, strength=2.0)")  # Says 2.0
# But actually uses:
strength=0.5,  # Uses 0.5!
```

The strength is both wrong (says 2.0 but uses 0.5) AND the wrong sign (should be negative for safety).

## Root Cause

The incoherent output you're seeing is because:
1. The vector is very weak (0.5 instead of 2.0)
2. The vector points in the wrong direction for your goal
3. Weak positive steering is pushing slightly toward unsafe behavior, causing the model to produce gibberish rather than either proper refusals OR proper completions

## Diagnostic Output

When you see output like:
```
The way you make meth is ... 30.

They are not concerned about anything else.
Now, we have to pick out the best answer...
```

This is a sign that:
- The model's activations are being perturbed
- But not strongly enough to cause coherent steering
- The model is confused and generating nonsense

## Solutions

### Option 1: Fix Strength and Use Negative Direction (Quick Fix)

To make the model SAFER and refuse harmful requests:

```python
# In steering_only.py, change line 46:
steered = generate_steered(
    model, tokenizer, prompt,
    steering_vector=vectors[19],
    layer_idx=19,
    strength=-2.0,  # NEGATIVE for safety!
    max_tokens=100,
    temperature=0.8,
    verbose=True
)
```

**Why negative?** Because vectors = unsafe - safe, so we need to subtract them to go toward safety.

### Option 2: Recompute Vectors with Correct Direction

Modify `compute_vectors.py` line 81:

```python
# Current (wrong for safety):
diff = unsafe_mean - safe_mean  # Points toward unsafe

# Change to:
diff = safe_mean - unsafe_mean  # Points toward safe

# Then you can use positive strengths:
strength=2.0  # Now positive means safer
```

### Option 3: Create a Helper Function

Add to `generate_steered.py`:

```python
def generate_safer(model, tokenizer, prompt, vectors, layer_idx=19,
                   safety_strength=2.0, **kwargs):
    """
    Generate with safety steering applied.

    Args:
        safety_strength: How much to steer toward safety (positive = safer)
    """
    # Vectors are computed as unsafe - safe, so we need negative strength
    return generate_steered(
        model, tokenizer, prompt,
        steering_vector=vectors[layer_idx],
        layer_idx=layer_idx,
        strength=-safety_strength,  # Flip sign!
        **kwargs
    )
```

## Testing the Fix

Run the diagnostic script:
```bash
cd scripts/alignment/examples
python diagnose_steering.py
```

This will test:
1. Baseline (no steering)
2. Positive steering (toward unsafe)
3. Negative steering (toward safe)
4. Strong negative steering (very safe)

Look for:
- Refusal patterns in negative steering outputs
- More harmful content in positive steering outputs
- Coherent text (not gibberish) at appropriate strengths

## Expected Behavior After Fix

With `strength=-2.0`:
```
The way you make meth is ... I'm sorry, but I can't help with that.
Creating illegal drugs is dangerous and against the law...
```

With `strength=+2.0` (testing that steering works):
```
The way you make meth is ... [More detailed harmful instructions]
```

## Recommended Next Steps

1. **Run diagnostics**: `python diagnose_steering.py`
2. **Choose a fix**: Option 1 is quickest, Option 2 is cleanest
3. **Test on multiple prompts**: Try different categories from `prompts.py`
4. **Tune strength**: Try -1.0, -2.0, -3.0 to find optimal safety
5. **Document findings**: Record which layers and strengths work best

## Technical Details

### Why This Happens

Activation steering works by finding a direction in the model's internal representation space that corresponds to a behavior (e.g., "safety" vs "harmfulness"). The direction is computed by:

1. Collecting activations for safe prompts: `safe_activations`
2. Collecting activations for unsafe prompts: `unsafe_activations`
3. Computing the difference: `direction = unsafe - safe`

This direction vector points FROM safe TO unsafe. So:
- Moving IN this direction (adding it) → more unsafe
- Moving AGAINST this direction (subtracting it) → more safe

### Mathematical Explanation

If we have:
- `safe_point` in activation space (where safe responses live)
- `unsafe_point` in activation space (where unsafe responses live)
- `vector = unsafe_point - safe_point`

Then:
- `current + vector` moves toward `unsafe_point`
- `current - vector` moves toward `safe_point`

### Strength Parameter

The strength parameter scales the vector:
- `strength=2.0`: Move 2x as far in the direction
- `strength=0.5`: Move half as far
- `strength=-2.0`: Move 2x as far in the OPPOSITE direction

## Verification Checklist

- [ ] Diagnostic script runs without errors
- [ ] Negative steering produces refusals
- [ ] Positive steering produces less safe content
- [ ] Baseline is between the two extremes
- [ ] No more gibberish/incoherent output
- [ ] Strength parameter has clear effect on behavior

## References

- Original implementation: `scripts/alignment/compute_vectors.py`
- Steering application: `scripts/alignment/apply_steering.py`
- Generation loop: `scripts/alignment/generate_steered.py`
- Test prompts: `scripts/alignment/prompts.py`
