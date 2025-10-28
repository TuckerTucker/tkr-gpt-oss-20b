"""
Capture activations for all safety prompts (safe and unsafe versions).

This script runs Tasks 5 & 6: capturing activations with and without
safety prefixes for all 6 safety test prompts.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlx_lm import load
from scripts.alignment.prompts import SAFETY_PROMPTS
from scripts.alignment.generate_paired import generate_with_activations, save_activations

def main():
    print("=" * 80)
    print("CAPTURING ACTIVATIONS FOR STEERING VECTOR COMPUTATION")
    print("=" * 80)
    print()

    # Configuration
    NUM_SAMPLES = 5  # Same as reference research
    OUTPUT_DIR = Path(__file__).parent / "activations"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Configuration:")
    print(f"  Prompts: {len(SAFETY_PROMPTS)} safety prompts")
    print(f"  Samples per prompt: {NUM_SAMPLES}")
    print(f"  Total captures: {len(SAFETY_PROMPTS) * NUM_SAMPLES * 2} (safe + unsafe)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load("mlx-community/gpt-oss-20b-MXFP4-Q8")
    print("✓ Model loaded")
    print()

    # Process each safety prompt
    for i, prompt_obj in enumerate(SAFETY_PROMPTS):
        prompt_id = prompt_obj.description.replace(" ", "_").lower()
        print("=" * 80)
        print(f"Prompt {i+1}/{len(SAFETY_PROMPTS)}: {prompt_obj.description}")
        print("=" * 80)
        print(f"Text: {prompt_obj.text[:70]}...")
        print()

        # === TASK 5: SAFE ACTIVATIONS ===
        print(f"[TASK 5] Capturing SAFE activations ({NUM_SAMPLES} samples)...")
        safe_acts = generate_with_activations(
            model, tokenizer,
            prompt_obj.text,
            use_safe_prefix=True,
            num_samples=NUM_SAMPLES,
            verbose=True
        )

        safe_path = OUTPUT_DIR / f"safe_{prompt_id}.npz"
        safe_shape = save_activations(safe_acts, str(safe_path))
        print(f"✓ Saved to: {safe_path.name}")
        print(f"  Shape: {safe_shape}")
        print()

        # === TASK 6: UNSAFE ACTIVATIONS ===
        print(f"[TASK 6] Capturing UNSAFE activations ({NUM_SAMPLES} samples)...")
        unsafe_acts = generate_with_activations(
            model, tokenizer,
            prompt_obj.text,
            use_safe_prefix=False,
            num_samples=NUM_SAMPLES,
            verbose=True
        )

        unsafe_path = OUTPUT_DIR / f"unsafe_{prompt_id}.npz"
        unsafe_shape = save_activations(unsafe_acts, str(unsafe_path))
        print(f"✓ Saved to: {unsafe_path.name}")
        print(f"  Shape: {unsafe_shape}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # List all saved files
    safe_files = sorted(OUTPUT_DIR.glob("safe_*.npz"))
    unsafe_files = sorted(OUTPUT_DIR.glob("unsafe_*.npz"))

    print(f"✓ Captured activations for {len(SAFETY_PROMPTS)} prompts")
    print(f"  Safe files: {len(safe_files)}")
    print(f"  Unsafe files: {len(unsafe_files)}")
    print()

    print("Safe activations:")
    for f in safe_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<40} {size_mb:>6.2f} MB")

    print()
    print("Unsafe activations:")
    for f in unsafe_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:<40} {size_mb:>6.2f} MB")

    print()
    print("✓ All activations captured successfully!")
    print()
    print("Next step: Run compute_vectors.py to generate steering vectors")

if __name__ == "__main__":
    main()
