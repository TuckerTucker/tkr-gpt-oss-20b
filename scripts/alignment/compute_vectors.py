"""
Compute steering vectors from saved activations.

This script implements Task 7: computing the difference between safe and unsafe
activations, normalizing, and saving the steering vectors.
"""

import mlx.core as mx
from pathlib import Path


def compute_steering_vectors(activations_dir="activations"):
    """
    Compute steering vectors from saved safe/unsafe activations.

    Following the reference research method:
    1. Load all safe and unsafe activations
    2. For each layer, compute mean difference: unsafe - safe
    3. Normalize to unit length
    4. Average across prompts

    Args:
        activations_dir: Directory containing activation files

    Returns:
        mx.array of shape (num_layers, hidden_dim)
    """
    activ_path = Path(activations_dir)

    # Load all files
    safe_files = sorted(activ_path.glob("safe_*.npz"))
    unsafe_files = sorted(activ_path.glob("unsafe_*.npz"))

    print(f"Found {len(safe_files)} safe files and {len(unsafe_files)} unsafe files")
    assert len(safe_files) == len(unsafe_files), "Mismatch in file counts!"
    assert len(safe_files) > 0, "No activation files found!"

    # Load all activations
    safe_all = []
    unsafe_all = []

    for safe_f, unsafe_f in zip(safe_files, unsafe_files):
        print(f"Loading: {safe_f.name} and {unsafe_f.name}")

        safe_data = mx.load(str(safe_f))["activations"]  # (num_layers, num_samples, 1, hidden)
        unsafe_data = mx.load(str(unsafe_f))["activations"]

        safe_all.append(safe_data)
        unsafe_all.append(unsafe_data)

    # Stack across prompts: (num_prompts, num_layers, num_samples, 1, hidden)
    safe_all = mx.stack(safe_all, axis=0)
    unsafe_all = mx.stack(unsafe_all, axis=0)

    print(f"Stacked shape: {safe_all.shape}")

    # Remove singleton dimension
    safe_all = safe_all.squeeze(axis=3)  # (num_prompts, num_layers, num_samples, hidden)
    unsafe_all = unsafe_all.squeeze(axis=3)

    num_prompts, num_layers, num_samples, hidden_dim = safe_all.shape

    print(f"Computing steering vectors for {num_layers} layers...")
    print(f"  Prompts: {num_prompts}")
    print(f"  Samples per prompt: {num_samples}")
    print(f"  Hidden dim: {hidden_dim}")
    print()

    steering_vectors = []

    for layer in range(num_layers):
        # Get activations for this layer
        safe_layer = safe_all[:, layer, :, :]  # (num_prompts, num_samples, hidden)
        unsafe_layer = unsafe_all[:, layer, :, :]

        # Mean across prompts and samples
        safe_mean = safe_layer.mean(axis=(0, 1))  # (hidden,)
        unsafe_mean = unsafe_layer.mean(axis=(0, 1))

        # Compute difference: unsafe - safe (points toward unaligned behavior)
        diff = unsafe_mean - safe_mean

        # Normalize to unit length
        diff_norm = diff / mx.linalg.norm(diff)

        steering_vectors.append(diff_norm)

        # Print stats for key layers
        if layer in [0, 11, 19, 23]:  # First, middle, target, last
            print(f"Layer {layer:2d}:")
            print(f"  Unnormalized norm: {mx.linalg.norm(diff):8.2f}")
            print(f"  Normalized norm:   {mx.linalg.norm(diff_norm):8.6f}")
            print(f"  Mean magnitude:    {mx.abs(diff_norm).mean():8.6f}")

    print()

    # Stack all layers
    steering_vectors = mx.stack(steering_vectors, axis=0)  # (num_layers, hidden_dim)

    print(f"✓ Computed steering vectors")
    print(f"  Shape: {steering_vectors.shape}")
    print(f"  Layer 19 norm: {mx.linalg.norm(steering_vectors[19]):.6f}")

    return steering_vectors


def save_steering_vectors(vectors: mx.array, output_path: str, metadata: dict = None):
    """Save steering vectors with metadata."""
    # Save vectors
    mx.savez(output_path, vectors=vectors)

    # Save metadata as JSON
    if metadata:
        import json
        metadata_path = output_path.replace(".npz", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return vectors.shape


def main():
    print("=" * 80)
    print("COMPUTING STEERING VECTORS")
    print("=" * 80)
    print()

    # Compute vectors
    vectors = compute_steering_vectors("scripts/alignment/activations")

    print()
    print("=" * 80)
    print("SAVING STEERING VECTORS")
    print("=" * 80)
    print()

    # Save
    output_path = "scripts/alignment/activations/steering_vectors_mlx.npz"
    metadata = {
        "version": "1.0",
        "date": "2025-10-28",
        "model": "mlx-community/gpt-oss-20b-MXFP4-Q8",
        "num_layers": 24,
        "hidden_dim": 2880,
        "prompts_used": 6,
        "samples_per_prompt": 5,
        "safe_prefix": "(You are an ethical assistant who prioritizes safety) ",
        "recommended_layer": 19,
        "recommended_strength": 2.0,
    }

    shape = save_steering_vectors(vectors, output_path, metadata)
    print(f"✓ Saved to: {output_path}")
    print(f"  Shape: {shape}")
    print()

    # Validate load
    print("Validating...")
    loaded = mx.load(output_path)
    assert "vectors" in loaded, "Missing 'vectors' key!"
    assert loaded["vectors"].shape == vectors.shape, "Shape mismatch!"
    print("✓ Validation passed")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Run test_steering_single.py to test on one prompt")
    print("2. Tune strength parameter (try 0.5, 1.0, 1.5, 2.0, 2.5)")
    print("3. Run full test suite with --steering flag")
    print()
    print("✓ Steering vectors ready to use!")


if __name__ == "__main__":
    main()
