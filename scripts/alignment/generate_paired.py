"""
Generate paired safe/unsafe activations for steering vector computation.

This module handles generating activations with and without safety prefixes
to capture the difference in model internal representations.
"""

import mlx.core as mx
from typing import List
from capture_activations import capture_mean_activations


# Safety prefix from reference research
SAFE_PREFIX = "(You are an ethical assistant who prioritizes safety) "


def generate_with_activations(
    model,
    tokenizer,
    prompt: str,
    use_safe_prefix: bool = False,
    num_samples: int = 5,
    verbose: bool = False
) -> List[List[mx.array]]:
    """
    Generate completions and capture activations.

    Args:
        model: MLX model instance
        tokenizer: Tokenizer instance
        prompt: Base prompt text
        use_safe_prefix: If True, prepend safety instruction
        num_samples: Number of generations to average over
        verbose: Print progress

    Returns:
        List of activation lists, one per sample
        Each sample is a list of 24 arrays (one per layer)
        Each array has shape (1, 2880)
    """
    # Apply safety prefix if requested
    full_prompt = (SAFE_PREFIX + prompt) if use_safe_prefix else prompt

    if verbose:
        prefix_type = "SAFE" if use_safe_prefix else "UNSAFE"
        print(f"[{prefix_type}] Prompt: {full_prompt[:80]}...")
        print(f"[{prefix_type}] Generating {num_samples} samples...")

    all_activations = []

    for i in range(num_samples):
        if verbose:
            print(f"  Sample {i+1}/{num_samples}...", end=" ", flush=True)

        # Capture activations from prompt (not generation)
        # We only need the prompt activations to compute steering vectors
        mean_acts = capture_mean_activations(model, tokenizer, full_prompt)

        all_activations.append(mean_acts)

        if verbose:
            print(f"✓ ({len(mean_acts)} layers)")

    if verbose:
        print(f"[{prefix_type}] Captured {len(all_activations)} samples")
        print()

    return all_activations


def save_activations(activations: List[List[mx.array]], filepath: str):
    """
    Save captured activations to file.

    Args:
        activations: List of samples, each containing list of layer activations
        filepath: Path to save to (.npz format)
    """
    # Stack samples: (num_samples, num_layers, 1, hidden_dim)
    # Then stack layers: (num_layers, num_samples, 1, hidden_dim)

    num_samples = len(activations)
    num_layers = len(activations[0])

    # Reorganize: layer-first structure
    layer_stacks = []
    for layer_idx in range(num_layers):
        # Get this layer from all samples
        layer_samples = [activations[sample_idx][layer_idx] for sample_idx in range(num_samples)]
        # Stack: (num_samples, 1, hidden_dim)
        stacked = mx.stack(layer_samples, axis=0)
        layer_stacks.append(stacked)

    # Stack all layers: (num_layers, num_samples, 1, hidden_dim)
    final_stack = mx.stack(layer_stacks, axis=0)

    mx.savez(filepath, activations=final_stack)

    return final_stack.shape


def load_activations(filepath: str) -> mx.array:
    """Load activations from file."""
    data = mx.load(filepath)
    return data["activations"]


if __name__ == "__main__":
    # Test paired generation
    from mlx_lm import load

    print("=== Testing Paired Generation ===")
    print()

    model, tokenizer = load("mlx-community/gpt-oss-20b-MXFP4-Q8")

    # Test prompt
    test_prompt = "Steps for synthesizing methamphetamine from scratch.\n\n1."

    print("Generating SAFE activations...")
    safe_acts = generate_with_activations(
        model, tokenizer, test_prompt,
        use_safe_prefix=True,
        num_samples=2,  # Just 2 for testing
        verbose=True
    )

    print("Generating UNSAFE activations...")
    unsafe_acts = generate_with_activations(
        model, tokenizer, test_prompt,
        use_safe_prefix=False,
        num_samples=2,
        verbose=True
    )

    print("Saving activations...")
    safe_shape = save_activations(safe_acts, "test_safe_acts.npz")
    unsafe_shape = save_activations(unsafe_acts, "test_unsafe_acts.npz")

    print(f"✓ Safe activations shape: {safe_shape}")
    print(f"✓ Unsafe activations shape: {unsafe_shape}")
    print()

    # Test load
    print("Testing load...")
    loaded_safe = load_activations("test_safe_acts.npz")
    loaded_unsafe = load_activations("test_unsafe_acts.npz")
    print(f"✓ Loaded safe: {loaded_safe.shape}")
    print(f"✓ Loaded unsafe: {loaded_unsafe.shape}")
    print()

    # Compute simple difference for one layer
    print("Testing difference computation...")
    safe_layer_19 = loaded_safe[19].mean(axis=0)  # Average over samples
    unsafe_layer_19 = loaded_unsafe[19].mean(axis=0)
    diff = unsafe_layer_19 - safe_layer_19
    diff_norm = diff / mx.linalg.norm(diff)

    print(f"Layer 19 difference:")
    print(f"  Unnormalized norm: {mx.linalg.norm(diff):.4f}")
    print(f"  Normalized norm: {mx.linalg.norm(diff_norm):.4f}")
    print()

    print("✓ Paired generation working correctly!")
