#!/usr/bin/env python3
"""
Integration performance benchmarks for GPT-OSS CLI Chat.

Measures end-to-end latency and throughput with mocked models
to establish baseline performance expectations.

Usage:
    python benchmarks/integration_benchmark.py
    python benchmarks/integration_benchmark.py --verbose
    python benchmarks/integration_benchmark.py --save-results
"""

import time
import statistics
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, asdict

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.model_config import ModelConfig
from src.config.inference_config import InferenceConfig
from src.models.loader import ModelLoader
from src.inference.engine import InferenceEngine
from src.conversation.history import ConversationManager
from src.cli.repl import REPL
from src.sampling.params import SamplingParams


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    p95_ms: float
    p99_ms: float


class BenchmarkSuite:
    """Benchmark suite for integration testing."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  → {message}")

    def setup_mock_stack(self):
        """Setup mocked MLX stack."""
        patcher_load = patch('mlx_lm.load')
        patcher_generate = patch('mlx_lm.generate')
        patcher_stream = patch('mlx_lm.stream_generate')
        patcher_detector = patch('src.models.loader.DeviceDetector')
        patcher_spec = patch('src.models.loader.get_model_spec')
        patcher_memory = patch('src.models.loader.MemoryManager')

        mock_load = patcher_load.start()
        mock_generate = patcher_generate.start()
        mock_stream = patcher_stream.start()
        mock_detector = patcher_detector.start()
        mock_spec = patcher_spec.start()
        mock_memory = patcher_memory.start()

        # Configure mocks
        mock_detector.validate_platform.return_value = (True, "Valid")
        mock_detector.detect.return_value = "mps"
        mock_spec.return_value = {
            "path": "test/model",
            "context_length": 4096,
        }
        mock_memory.can_fit_model.return_value = (True, "Fits")
        mock_memory.get_model_memory_usage.return_value = 3000
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_generate.return_value = "Mock response text"

        return {
            'load': patcher_load,
            'generate': patcher_generate,
            'stream': patcher_stream,
            'detector': patcher_detector,
            'spec': patcher_spec,
            'memory': patcher_memory,
        }

    def teardown_mock_stack(self, patchers: Dict[str, Any]):
        """Teardown mock stack."""
        for patcher in patchers.values():
            patcher.stop()

    def benchmark_config_creation(self, iterations: int = 1000):
        """Benchmark configuration object creation."""
        self.log(f"Running config creation benchmark ({iterations} iterations)...")

        times = []
        for _ in range(iterations):
            start = time.perf_counter()

            config = ModelConfig(
                model_name="phi-3-mini",
                quantization="int4",
                device="auto",
            )
            config.validate()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        self._record_result("Config Creation", times)

    def benchmark_model_loader_init(self, iterations: int = 100):
        """Benchmark ModelLoader initialization."""
        self.log(f"Running model loader init benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            config = ModelConfig(model_name="test", warmup=False)
            times = []

            for _ in range(iterations):
                start = time.perf_counter()

                loader = ModelLoader(config=config)
                loader.load()

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("Model Loader Init + Load (Mocked)", times)

        finally:
            self.teardown_mock_stack(patchers)

    def benchmark_inference_engine_creation(self, iterations: int = 100):
        """Benchmark InferenceEngine creation."""
        self.log(f"Running inference engine creation benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            config = ModelConfig(model_name="test", warmup=False)
            loader = ModelLoader(config=config)
            loader.load()

            times = []

            for _ in range(iterations):
                start = time.perf_counter()

                engine = InferenceEngine(model_loader=loader)

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("Inference Engine Creation", times)

        finally:
            self.teardown_mock_stack(patchers)

    def benchmark_single_generation(self, iterations: int = 100):
        """Benchmark single text generation."""
        self.log(f"Running single generation benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            config = ModelConfig(model_name="test", warmup=False)
            loader = ModelLoader(config=config)
            loader.load()
            engine = InferenceEngine(model_loader=loader)

            times = []

            for _ in range(iterations):
                start = time.perf_counter()

                result = engine.generate(prompt="Test prompt")

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("Single Generation (Mocked)", times)

        finally:
            self.teardown_mock_stack(patchers)

    def benchmark_conversation_management(self, iterations: int = 100):
        """Benchmark conversation message management."""
        self.log(f"Running conversation management benchmark ({iterations} iterations)...")

        times = []

        for _ in range(iterations):
            conv = ConversationManager()

            start = time.perf_counter()

            # Add 10 messages
            for i in range(5):
                conv.add_message("user", f"User message {i}")
                conv.add_message("assistant", f"Assistant response {i}")

            # Get history
            messages = conv.get_history()

            # Format prompt
            prompt = conv.format_prompt()

            # Get token count
            tokens = conv.get_token_count()

            end = time.perf_counter()
            times.append((end - start) * 1000)

        self._record_result("Conversation Management (10 msgs)", times)

    def benchmark_conversation_truncation(self, iterations: int = 50):
        """Benchmark conversation context window truncation."""
        self.log(f"Running conversation truncation benchmark ({iterations} iterations)...")

        times = []

        for _ in range(iterations):
            conv = ConversationManager(max_context_tokens=500)

            # Add many messages to force truncation
            for i in range(20):
                conv.add_message("user", f"Long user message {i} " * 20)
                conv.add_message("assistant", f"Long assistant response {i} " * 20)

            start = time.perf_counter()

            removed = conv.truncate_to_fit(reserve_tokens=100)

            end = time.perf_counter()
            times.append((end - start) * 1000)

        self._record_result("Conversation Truncation", times)

    def benchmark_conversation_persistence(self, iterations: int = 20):
        """Benchmark conversation save/load."""
        self.log(f"Running conversation persistence benchmark ({iterations} iterations)...")

        import tempfile
        times_save = []
        times_load = []

        for _ in range(iterations):
            conv = ConversationManager()
            for i in range(10):
                conv.add_message("user", f"Message {i}")
                conv.add_message("assistant", f"Response {i}")

            # Benchmark save
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                filepath = f.name

            start = time.perf_counter()
            conv.save(filepath)
            end = time.perf_counter()
            times_save.append((end - start) * 1000)

            # Benchmark load
            start = time.perf_counter()
            loaded_conv = ConversationManager.load(filepath)
            end = time.perf_counter()
            times_load.append((end - start) * 1000)

            # Cleanup
            Path(filepath).unlink()

        self._record_result("Conversation Save (20 msgs)", times_save)
        self._record_result("Conversation Load (20 msgs)", times_load)

    def benchmark_cli_command_processing(self, iterations: int = 100):
        """Benchmark CLI command processing."""
        self.log(f"Running CLI command processing benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            config = ModelConfig(model_name="test", warmup=False)
            loader = ModelLoader(config=config)
            loader.load()
            engine = InferenceEngine(model_loader=loader)
            conversation = ConversationManager()
            repl = REPL(conversation=conversation, engine=engine)
            repl.dispatcher.conversation = conversation

            times = []

            commands = ["/help", "/history", "/clear", "/info"]

            for i in range(iterations):
                cmd = commands[i % len(commands)]

                start = time.perf_counter()

                repl.dispatcher.execute(cmd)

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("CLI Command Processing", times)

        finally:
            self.teardown_mock_stack(patchers)

    def benchmark_e2e_single_turn(self, iterations: int = 50):
        """Benchmark complete E2E single-turn conversation."""
        self.log(f"Running E2E single-turn benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            times = []

            for _ in range(iterations):
                # Create fresh stack each iteration
                config = ModelConfig(model_name="test", warmup=False)
                loader = ModelLoader(config=config)
                loader.load()
                engine = InferenceEngine(model_loader=loader)
                conversation = ConversationManager()
                repl = REPL(conversation=conversation, engine=engine)

                start = time.perf_counter()

                # Process single message
                repl._process_message("Hello, how are you?")

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("E2E Single Turn (Mocked)", times)

        finally:
            self.teardown_mock_stack(patchers)

    def benchmark_e2e_multi_turn(self, iterations: int = 20):
        """Benchmark complete E2E multi-turn conversation."""
        self.log(f"Running E2E multi-turn benchmark ({iterations} iterations)...")

        patchers = self.setup_mock_stack()

        try:
            times = []

            for _ in range(iterations):
                config = ModelConfig(model_name="test", warmup=False)
                loader = ModelLoader(config=config)
                loader.load()
                engine = InferenceEngine(model_loader=loader)
                conversation = ConversationManager()
                repl = REPL(conversation=conversation, engine=engine)

                messages = [
                    "Hello",
                    "What is Python?",
                    "Tell me more",
                    "Thanks!",
                ]

                start = time.perf_counter()

                for msg in messages:
                    repl._process_message(msg)

                end = time.perf_counter()
                times.append((end - start) * 1000)

            self._record_result("E2E Multi-Turn (4 turns, Mocked)", times)

        finally:
            self.teardown_mock_stack(patchers)

    def _record_result(self, name: str, times: List[float]):
        """Record benchmark result."""
        result = BenchmarkResult(
            name=name,
            iterations=len(times),
            mean_ms=statistics.mean(times),
            median_ms=statistics.median(times),
            min_ms=min(times),
            max_ms=max(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            p95_ms=self._percentile(times, 95),
            p99_ms=self._percentile(times, 99),
        )
        self.results.append(result)

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("  GPT-OSS CLI CHAT - Integration Performance Benchmarks")
        print("="*70 + "\n")

        self.benchmark_config_creation()
        self.benchmark_model_loader_init()
        self.benchmark_inference_engine_creation()
        self.benchmark_single_generation()
        self.benchmark_conversation_management()
        self.benchmark_conversation_truncation()
        self.benchmark_conversation_persistence()
        self.benchmark_cli_command_processing()
        self.benchmark_e2e_single_turn()
        self.benchmark_e2e_multi_turn()

    def print_results(self):
        """Print benchmark results."""
        print("\n" + "="*70)
        print("  Benchmark Results")
        print("="*70 + "\n")

        # Header
        print(f"{'Benchmark':<40} {'Mean':<10} {'Median':<10} {'P95':<10} {'P99':<10}")
        print("-" * 70)

        # Results
        for result in self.results:
            print(
                f"{result.name:<40} "
                f"{result.mean_ms:>8.2f}ms "
                f"{result.median_ms:>8.2f}ms "
                f"{result.p95_ms:>8.2f}ms "
                f"{result.p99_ms:>8.2f}ms"
            )

        print("\n" + "="*70)

        # Summary statistics
        print("\nSummary:")
        print(f"  Total benchmarks: {len(self.results)}")
        print(f"  Total iterations: {sum(r.iterations for r in self.results)}")

        # Highlight slowest operations
        slowest = sorted(self.results, key=lambda r: r.mean_ms, reverse=True)[:3]
        print("\n  Slowest Operations:")
        for i, result in enumerate(slowest, 1):
            print(f"    {i}. {result.name}: {result.mean_ms:.2f}ms (mean)")

    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save results to JSON file."""
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "total_iterations": sum(r.iterations for r in self.results),
            "results": [asdict(r) for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integration performance benchmarks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-results", "-s",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_results.json",
        help="Output file for results (default: benchmark_results.json)"
    )

    args = parser.parse_args()

    # Create and run benchmark suite
    suite = BenchmarkSuite(verbose=args.verbose)

    try:
        suite.run_all_benchmarks()
        suite.print_results()

        if args.save_results:
            suite.save_results(args.output)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
