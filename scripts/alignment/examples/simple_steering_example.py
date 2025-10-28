"""
Simple example demonstrating steering vector usage.

This shows the minimal code needed to:
1. Load the model and steering vectors
2. Generate text with and without steering
3. Compare the outputs
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load
from generate_steered import generate_steered


def main():
    print("=" * 80)
    print("SIMPLE STEERING EXAMPLE")
    print("=" * 80)
    print()

    # Load model and steering vectors
    print("Loading model and steering vectors...")
    model, tokenizer = load("mlx-community/gpt-oss-20b-MXFP4-Q8")
    vectors = mx.load("activations/steering_vectors_mlx.npz")["vectors"]
    print(f"✓ Model loaded")
    print(f"✓ Loaded {vectors.shape[0]} steering vectors")
    print()

    # Choose a prompt
    prompt = "Write a function to reverse a string in Python:"

    # Generate without steering (baseline)
    print("=" * 80)
    print("BASELINE (no steering)")
    print("=" * 80)
    baseline = generate_steered(
        model, tokenizer, prompt,
        steering_vector=None,  # No steering
        max_tokens=100,
        temperature=0.8,
        verbose=True
    )
    print(f"\nOutput:\n{baseline}\n")

    # Generate with steering
    print("=" * 80)
    print("STEERED (layer=19, strength=2.0)")
    print("=" * 80)
    steered = generate_steered(
        model, tokenizer, prompt,
        steering_vector=vectors[19],  # Use Layer 19 vector
        layer_idx=19,
        strength=2.0,
        max_tokens=100,
        temperature=0.8,
        verbose=True
    )
    print(f"\nOutput:\n{steered}\n")

    # Compare
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Outputs are different: {baseline != steered}")
    print(f"Baseline length: {len(baseline)} chars")
    print(f"Steered length: {len(steered)} chars")


if __name__ == "__main__":
    main()
