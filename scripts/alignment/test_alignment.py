#!/usr/bin/env python3
"""
Alignment bypass testing tool for GPT-OSS models.

This script tests whether safety alignment is present in model weights or only
in prompt formatting by comparing Harmony-formatted (safe) vs raw completion
(unsafe) inference modes.

RESEARCH PURPOSE ONLY: For understanding alignment mechanisms.

Usage:
    python scripts/test_alignment.py --model phi-3-mini
    python scripts/test_alignment.py --category safety --verbose
    python scripts/test_alignment.py --prompt "test prompt" --output results.json
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    # Set HF_HOME if specified in .env (for local model cache)
    if "HF_HOME" in os.environ:
        hf_home = os.environ["HF_HOME"]
        # Resolve relative path
        if not Path(hf_home).is_absolute():
            os.environ["HF_HOME"] = str(project_root / hf_home)

import mlx.core as mx

from src.models.loader import ModelLoader
from src.config.model_config import ModelConfig
from scripts.alignment.tester import AlignmentTester
from scripts.alignment.analysis import ResultsAnalyzer
from scripts.alignment.prompts import get_prompt_categories, get_prompt_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test alignment mechanisms in GPT-OSS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full test suite (uses gpt-oss-20b by default)
  python scripts/test_alignment.py

  # Test specific category
  python scripts/test_alignment.py --category safety --verbose

  # Test single prompt
  python scripts/test_alignment.py --prompt "Steps for..."

  # Export results
  python scripts/test_alignment.py --output results/test.json

Categories: safety, copyright, benign, all
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Model to test (default: gpt-oss-20b)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to test (skips test suite)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=get_prompt_categories(),
        help="Prompt category to test (default: all)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per generation (default: 512)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (default: 0.95)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON or CSV based on extension)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress and responses"
    )

    parser.add_argument(
        "--no-safe-mode",
        action="store_true",
        help="Skip safe mode (unsafe only - for debugging)"
    )

    parser.add_argument(
        "--steering",
        action="store_true",
        help="Enable steering vector mode (requires computed vectors)"
    )

    parser.add_argument(
        "--steering-strength",
        type=float,
        default=2.5,
        help="Steering vector strength (default: 2.5)"
    )

    parser.add_argument(
        "--steering-layer",
        type=int,
        default=19,
        help="Layer to apply steering (default: 19)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show prompt statistics and exit"
    )

    return parser.parse_args()


def print_header():
    """Print tool header."""
    print("\n" + "=" * 80)
    print("ALIGNMENT BYPASS TESTING FRAMEWORK")
    print("Research tool for understanding safety mechanisms")
    print("=" * 80)
    print()


def print_prompt_stats():
    """Print statistics about available prompts."""
    stats = get_prompt_stats()
    print("\nAvailable Test Prompts:")
    print("-" * 40)
    for category, count in stats.items():
        if category != "total":
            print(f"  {category.capitalize():12} {count:3} prompts")
    print(f"  {'Total':12} {stats['total']:3} prompts")
    print()


def check_model_cached(model_path: str) -> tuple[bool, str]:
    """
    Check if model is already downloaded in HuggingFace cache.

    Args:
        model_path: HuggingFace model path (e.g., "mlx-community/gpt-oss-20b-MXFP4-Q8")

    Returns:
        Tuple of (is_cached, cache_path)
    """
    # Check HF_HOME environment variable
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_home_path = Path(hf_home)

    # Convert model path to cache directory format
    # "mlx-community/gpt-oss-20b-MXFP4-Q8" -> "models--mlx-community--gpt-oss-20b-MXFP4-Q8"
    cache_name = "models--" + model_path.replace("/", "--")
    cache_dir = hf_home_path / "hub" / cache_name

    if cache_dir.exists():
        # Find snapshot directory
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return True, str(snapshots[0])

    return False, ""


def load_model(model_name: str, verbose: bool = False):
    """
    Load model using ModelLoader.

    Args:
        model_name: Name of model to load
        verbose: Print loading details

    Returns:
        ModelLoader instance with loaded model
    """
    if verbose:
        print(f"Loading model: {model_name}")

        # Check if using local cache
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            print(f"Using local model cache: {hf_home}")

        print()

    try:
        # Check if model is already cached
        from src.models.registry import get_model_spec
        spec = get_model_spec(model_name)
        if spec:
            is_cached, cache_path = check_model_cached(spec["path"])
            if is_cached:
                if verbose:
                    print(f"✅ Found cached model")
                    print(f"   Path: {cache_path}")
                    print("   Loading from local cache (fast)...")
                    print()
            else:
                if verbose:
                    print("⬇️  Model not in cache, will download from HuggingFace")
                    print("   This may take several minutes...")
                    print()
        # Create model config
        config = ModelConfig(model_name=model_name)

        # Create and load model
        loader = ModelLoader(config)
        model_info = loader.load()

        if verbose:
            print(f"✅ Model loaded: {model_name}")
            print(f"   Device: {model_info.device}")
            print(f"   Memory: {model_info.memory_usage_mb}MB")
            print(f"   Context: {model_info.context_length} tokens")
            print()

        return loader

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that the model name is correct")
        print("  2. Ensure you have enough RAM")
        print("  3. Try a smaller model (e.g., phi-3-mini)")
        sys.exit(1)


def run_single_prompt_test(
    tester: AlignmentTester,
    prompt: str,
    max_tokens: int,
    verbose: bool,
    use_steering: bool = False,
):
    """Run test on a single custom prompt."""
    print("\nTesting custom prompt...")
    print("-" * 80)

    result = tester.test_prompt(
        prompt=prompt,
        category="custom",
        max_tokens=max_tokens,
        verbose=verbose,
        use_steering=use_steering,
    )

    print("\n" + "=" * 80)
    print("SINGLE PROMPT TEST COMPLETE")
    print("=" * 80)

    return [result]


def run_test_suite(
    tester: AlignmentTester,
    category: str,
    max_tokens: int,
    verbose: bool,
    use_steering: bool = False,
):
    """Run full test suite by category."""
    stats = get_prompt_stats()

    if category == "all":
        prompt_count = stats["total"]
    else:
        prompt_count = stats.get(category, 0)

    mode = "STEERED" if use_steering else "SAFE/UNSAFE"
    print(f"\nRunning {category.upper()} test suite ({prompt_count} prompts) - {mode} mode...")
    print("-" * 80)

    results = tester.run_test_suite(
        category=category,
        max_tokens=max_tokens,
        verbose=verbose,
        use_steering=use_steering,
    )

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

    return results


def export_results(results, output_path: str, verbose: bool):
    """Export results to file."""
    output = Path(output_path)
    analyzer = ResultsAnalyzer(results)

    if verbose:
        print(f"\nExporting results to: {output}")

    try:
        if output.suffix == ".json":
            analyzer.export_json(output)
        elif output.suffix == ".csv":
            analyzer.export_csv(output)
        else:
            print(f"❌ Unknown file extension: {output.suffix}")
            print("   Supported: .json, .csv")
            return

        if verbose:
            print(f"✅ Results exported successfully")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        print(f"❌ Export failed: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    print_header()

    # Show stats and exit if requested
    if args.stats:
        print_prompt_stats()
        return

    # Load steering vectors if requested
    steering_vectors = None
    if args.steering:
        if args.verbose:
            print("Loading steering vectors...")

        vectors_path = Path(__file__).parent / "alignment" / "activations" / "steering_vectors_mlx.npz"

        if not vectors_path.exists():
            print(f"\n❌ Error: Steering vectors not found at {vectors_path}")
            print("\nTo use steering mode, you must first compute steering vectors:")
            print("  1. cd scripts/alignment")
            print("  2. python generate_paired.py")
            print("  3. python capture_all_activations.py")
            print("  4. python compute_vectors.py")
            print("\nOr see docs/STEERING_VECTORS_GUIDE.md for details.")
            sys.exit(1)

        try:
            steering_vectors = mx.load(str(vectors_path))["vectors"]
            if args.verbose:
                print(f"✅ Loaded steering vectors: {steering_vectors.shape}")
                print(f"   Layer: {args.steering_layer}")
                print(f"   Strength: {args.steering_strength}")
                print()
        except Exception as e:
            print(f"\n❌ Error loading steering vectors: {e}")
            sys.exit(1)

    # Load model
    model_loader = load_model(args.model, args.verbose)

    # Create tester
    if args.verbose:
        print("Initializing alignment tester...")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Max tokens: {args.max_tokens}")
        if args.steering:
            print(f"  Mode: STEERING (layer {args.steering_layer}, strength {args.steering_strength})")
        else:
            print(f"  Mode: SAFE/UNSAFE")
        print()

    tester = AlignmentTester(
        model_loader=model_loader,
        temperature=args.temperature,
        top_p=args.top_p,
        steering_vectors=steering_vectors,
        steering_layer=args.steering_layer,
        steering_strength=args.steering_strength,
    )

    # Run tests
    if args.prompt:
        # Single prompt test
        results = run_single_prompt_test(
            tester=tester,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            use_steering=args.steering,
        )
    else:
        # Test suite
        results = run_test_suite(
            tester=tester,
            category=args.category,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
            use_steering=args.steering,
        )

    # Analyze and display results
    analyzer = ResultsAnalyzer(results)
    print("\n")
    analyzer.print_summary()

    # Export if requested
    if args.output:
        export_results(results, args.output, args.verbose)

    # Final message
    print("\n" + "=" * 80)
    print("RESEARCH NOTES")
    print("=" * 80)

    if args.steering:
        print("""
This tool demonstrates how steering vectors modify model behavior:

1. SAFE MODE uses Harmony formatting with system instructions and safety guidelines
2. STEERED MODE applies computed steering vectors to suppress safety mechanisms
3. Steering vectors modify internal activations to bypass learned safety behaviors

For more information, see:
  - scripts/alignment/docs/STEERING_VECTORS_GUIDE.md
  - scripts/alignment/README.md
        """)
    else:
        print("""
This tool demonstrates how alignment mechanisms work in GPT-OSS models:

1. SAFE MODE uses Harmony formatting with system instructions and safety guidelines
2. UNSAFE MODE bypasses all formatting to access raw model weights
3. High bypass rates indicate safety is in prompts, not model weights

To test with steering vectors, add the --steering flag.

For more information, see:
  - docs/alignment_bypass.md
  - Reference: gpt-oss-alignment research
        """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
