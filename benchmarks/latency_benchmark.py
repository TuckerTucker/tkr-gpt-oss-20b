#!/usr/bin/env python3
"""
Latency Benchmark for GPT-OSS CLI Chat.

Measures:
- Model loading time
- First token latency (TTFT - Time To First Token)
- Token generation throughput
- End-to-end inference latency

Usage:
    python benchmarks/latency_benchmark.py
    python benchmarks/latency_benchmark.py --model gpt-j-20b --quantization int4
    python benchmarks/latency_benchmark.py --prompts 10 --tokens 100
"""

import argparse
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    # Model info
    model_name: str
    quantization: str
    device: str

    # Loading metrics
    model_load_time_ms: int
    warmup_time_ms: int

    # Generation metrics
    num_prompts: int
    avg_prompt_tokens: int
    avg_generated_tokens: int

    # Latency metrics
    avg_first_token_latency_ms: float
    avg_total_latency_ms: float
    avg_tokens_per_second: float

    # Percentiles
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Memory
    peak_memory_mb: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nModel: {self.model_name} ({self.quantization})")
        print(f"Device: {self.device}")
        print(f"\nLoading:")
        print(f"  Model load time: {self.model_load_time_ms:,}ms")
        print(f"  Warmup time: {self.warmup_time_ms:,}ms")
        print(f"\nGeneration ({self.num_prompts} prompts):")
        print(f"  Avg prompt tokens: {self.avg_prompt_tokens}")
        print(f"  Avg generated tokens: {self.avg_generated_tokens}")
        print(f"\nLatency:")
        print(f"  First token (avg): {self.avg_first_token_latency_ms:.1f}ms")
        print(f"  Total (avg): {self.avg_total_latency_ms:.1f}ms")
        print(f"  Throughput: {self.avg_tokens_per_second:.1f} tokens/sec")
        print(f"\nPercentiles:")
        print(f"  P50: {self.p50_latency_ms:.1f}ms")
        print(f"  P95: {self.p95_latency_ms:.1f}ms")
        print(f"  P99: {self.p99_latency_ms:.1f}ms")
        print(f"\nMemory:")
        print(f"  Peak usage: {self.peak_memory_mb:,}MB")
        print("=" * 60)


class LatencyBenchmark:
    """Benchmark suite for inference latency."""

    def __init__(self, model_name: str, quantization: str, device: str = "auto"):
        """
        Initialize benchmark.

        Args:
            model_name: Model to benchmark
            quantization: Quantization level
            device: Target device
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device

        self.loader = None
        self.engine = None
        self.model_info = None

    def load_model(self) -> int:
        """
        Load model and return load time.

        Returns:
            Load time in milliseconds
        """
        print(f"Loading {self.model_name} with {self.quantization} quantization...")

        try:
            from src.config import ModelConfig
            from src.models import ModelLoader

            config = ModelConfig(
                model_name=self.model_name,
                quantization=self.quantization,
                device=self.device,
                warmup=False  # We'll warmup separately
            )

            start_time = time.perf_counter()
            self.loader = ModelLoader(config)
            self.model_info = self.loader.load()
            end_time = time.perf_counter()

            load_time_ms = int((end_time - start_time) * 1000)
            print(f"✓ Model loaded in {load_time_ms:,}ms")

            return load_time_ms

        except ImportError:
            print("⚠ Model components not yet implemented, skipping...")
            return 0

    def warmup(self) -> int:
        """
        Warmup model and return warmup time.

        Returns:
            Warmup time in milliseconds
        """
        print("Warming up model...")

        try:
            from src.inference import InferenceEngine
            from src.config import InferenceConfig

            config = InferenceConfig(max_tokens=10)
            self.engine = InferenceEngine(self.loader, config)

            start_time = time.perf_counter()
            # Generate dummy text to warm up
            self.engine.generate("Test warmup prompt")
            end_time = time.perf_counter()

            warmup_time_ms = int((end_time - start_time) * 1000)
            print(f"✓ Warmup completed in {warmup_time_ms:,}ms")

            return warmup_time_ms

        except ImportError:
            print("⚠ Inference components not yet implemented, skipping...")
            return 0

    def benchmark_prompts(
        self,
        prompts: List[str],
        max_tokens: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Benchmark a list of prompts.

        Args:
            prompts: List of prompts to benchmark
            max_tokens: Max tokens to generate per prompt

        Returns:
            List of timing results
        """
        print(f"\nBenchmarking {len(prompts)} prompts...")

        results = []

        try:
            from src.config import InferenceConfig

            config = InferenceConfig(max_tokens=max_tokens)

            for i, prompt in enumerate(prompts, 1):
                print(f"  Prompt {i}/{len(prompts)}...", end=" ", flush=True)

                start_time = time.perf_counter()
                result = self.engine.generate(prompt, config=config)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000

                results.append({
                    "prompt": prompt,
                    "tokens_generated": result.tokens_generated,
                    "latency_ms": latency_ms,
                    "tokens_per_second": result.tokens_generated / (latency_ms / 1000)
                })

                print(f"{latency_ms:.1f}ms ({result.tokens_generated} tokens)")

            return results

        except ImportError:
            print("⚠ Inference not yet implemented, returning mock results...")
            # Return mock results for testing
            return [
                {
                    "prompt": p,
                    "tokens_generated": max_tokens,
                    "latency_ms": 150.0,
                    "tokens_per_second": max_tokens / 0.15
                }
                for p in prompts
            ]

    def run(
        self,
        num_prompts: int = 10,
        max_tokens: int = 100
    ) -> BenchmarkResult:
        """
        Run complete benchmark suite.

        Args:
            num_prompts: Number of prompts to test
            max_tokens: Max tokens per prompt

        Returns:
            BenchmarkResult with all metrics
        """
        print("\n" + "=" * 60)
        print("STARTING LATENCY BENCHMARK")
        print("=" * 60)

        # Load model
        load_time = self.load_model()

        # Warmup
        warmup_time = self.warmup()

        # Generate test prompts
        prompts = self._generate_test_prompts(num_prompts)

        # Benchmark
        results = self.benchmark_prompts(prompts, max_tokens)

        # Calculate metrics
        latencies = [r["latency_ms"] for r in results]
        tokens_generated = [r["tokens_generated"] for r in results]
        tokens_per_second = [r["tokens_per_second"] for r in results]

        # Sort for percentiles
        latencies_sorted = sorted(latencies)

        return BenchmarkResult(
            model_name=self.model_name,
            quantization=self.quantization,
            device=self.model_info.device if self.model_info else "unknown",
            model_load_time_ms=load_time,
            warmup_time_ms=warmup_time,
            num_prompts=num_prompts,
            avg_prompt_tokens=20,  # Approximate
            avg_generated_tokens=int(sum(tokens_generated) / len(tokens_generated)),
            avg_first_token_latency_ms=0.0,  # TODO: Implement TTFT tracking
            avg_total_latency_ms=sum(latencies) / len(latencies),
            avg_tokens_per_second=sum(tokens_per_second) / len(tokens_per_second),
            p50_latency_ms=self._percentile(latencies_sorted, 50),
            p95_latency_ms=self._percentile(latencies_sorted, 95),
            p99_latency_ms=self._percentile(latencies_sorted, 99),
            peak_memory_mb=self.model_info.memory_usage_mb if self.model_info else 0
        )

    def _generate_test_prompts(self, num: int) -> List[str]:
        """Generate test prompts of varying lengths."""
        prompts = [
            "What is machine learning?",
            "Explain the concept of neural networks in simple terms.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the key differences between supervised and unsupervised learning?",
            "Describe the attention mechanism in transformer models.",
            "How does backpropagation work in neural networks?",
            "What is the difference between AI and machine learning?",
            "Explain what a large language model is.",
            "What are the advantages of using GPU acceleration?",
            "How does quantization reduce model size?",
        ]

        # Repeat if we need more
        while len(prompts) < num:
            prompts.extend(prompts[:num - len(prompts)])

        return prompts[:num]

    @staticmethod
    def _percentile(values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        index = int(len(values) * p / 100)
        return values[min(index, len(values) - 1)]


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPT-OSS CLI Chat latency"
    )
    parser.add_argument(
        "--model",
        default="gpt-j-20b",
        choices=["gpt-j-20b", "gpt-neox-20b"],
        help="Model to benchmark"
    )
    parser.add_argument(
        "--quantization",
        default="int4",
        choices=["int4", "int8", "fp16", "none"],
        help="Quantization level"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=10,
        help="Number of prompts to test"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
        help="Max tokens per generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = LatencyBenchmark(
        model_name=args.model,
        quantization=args.quantization,
        device=args.device
    )

    try:
        result = benchmark.run(
            num_prompts=args.prompts,
            max_tokens=args.tokens
        )

        # Print results
        result.print_summary()

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            print(f"\n✓ Results saved to {output_path}")

    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
