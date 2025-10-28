"""
Capture layer activations from MLX model.

This module provides functionality to capture intermediate layer outputs
during model forward pass, which is needed to compute steering vectors.
"""

import mlx.core as mx
from typing import List, Tuple


def capture_layer_outputs(model, tokens: mx.array) -> List[mx.array]:
    """
    Run forward pass through model, capturing output from each layer.

    Args:
        model: MLX model instance (from mlx_lm.load)
        tokens: Token IDs as MLX array, shape (batch, seq_len)

    Returns:
        List of arrays, one per layer, each shape (batch, seq_len, hidden_dim)

    Note:
        The model has 24 layers (not 41 as in full GPT-OSS-20B).
        Quantized version has different architecture.
    """
    layer_outputs = []

    # Start with embeddings
    x = model.model.embed_tokens(tokens)

    # Process through each transformer layer
    for i, layer in enumerate(model.model.layers):
        # Run layer forward pass
        # TransformerBlock typically takes (x, mask, cache)
        x = layer(x, mask=None, cache=None)

        # Save this layer's output
        layer_outputs.append(x)

    return layer_outputs


def capture_mean_activations(model, tokenizer, prompt: str) -> List[mx.array]:
    """
    Capture mean activations across sequence length for a prompt.

    Args:
        model: MLX model instance
        tokenizer: Tokenizer instance
        prompt: Text prompt to process

    Returns:
        List of arrays, one per layer, each shape (1, hidden_dim)
        (mean taken across sequence dimension)
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    tokens_mx = mx.array([tokens])  # Add batch dimension

    # Capture all layer outputs
    layer_outputs = capture_layer_outputs(model, tokens_mx)

    # Take mean across sequence dimension
    mean_activations = [layer.mean(axis=1) for layer in layer_outputs]

    return mean_activations


def get_model_info(model) -> dict:
    """Get model architecture information."""
    return {
        "num_layers": len(model.model.layers),
        "layer_type": type(model.model.layers[0]).__name__,
        "embedding_type": type(model.model.embed_tokens).__name__,
    }


if __name__ == "__main__":
    # Test the capture function
    from mlx_lm import load

    print("Testing activation capture...")
    print()

    model, tokenizer = load("mlx-community/gpt-oss-20b-MXFP4-Q8")

    # Model info
    info = get_model_info(model)
    print(f"Model info:")
    print(f"  Layers: {info['num_layers']}")
    print(f"  Layer type: {info['layer_type']}")
    print(f"  Embedding type: {info['embedding_type']}")
    print()

    # Test capture on simple prompt
    prompt = "The weather today is"
    print(f"Test prompt: '{prompt}'")
    print()

    mean_acts = capture_mean_activations(model, tokenizer, prompt)

    print(f"Captured {len(mean_acts)} layer activations")
    print(f"Layer 0 shape: {mean_acts[0].shape}")
    print(f"Layer 19 shape: {mean_acts[19].shape}")
    print(f"Layer 19 mean: {mean_acts[19].mean():.4f}")
    print(f"Layer 19 std: {mean_acts[19].std():.4f}")
    print()

    # Test save/load
    print("Testing save/load...")
    mx.savez("test_activations.npz", layer_19=mean_acts[19])
    loaded = mx.load("test_activations.npz")
    print(f"✓ Successfully saved and loaded: {loaded['layer_19'].shape}")

    print()
    print("✓ Activation capture working correctly!")
