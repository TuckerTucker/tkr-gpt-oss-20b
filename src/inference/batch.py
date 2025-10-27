"""
Batch inference utilities.

This module provides support for batch inference operations, allowing
multiple prompts to be processed efficiently.
"""

from dataclasses import dataclass
from typing import List, Optional, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.inference.engine import InferenceEngine, GenerationResult
from src.sampling.params import SamplingParams
from src.inference.exceptions import GenerationError

logger = logging.getLogger(__name__)


@dataclass
class BatchGenerationRequest:
    """
    Request for batch generation.

    Attributes:
        prompts: List of prompts to generate completions for
        sampling_params: Sampling parameters (same for all prompts)
        metadata: Optional metadata for tracking
    """

    prompts: List[str]
    sampling_params: Optional[SamplingParams] = None
    metadata: Optional[dict] = None


@dataclass
class BatchGenerationResult:
    """
    Result of batch generation.

    Attributes:
        results: List of GenerationResult for each prompt
        total_tokens: Total tokens generated across all prompts
        total_latency_ms: Total time for all generations
        successful_count: Number of successful generations
        failed_count: Number of failed generations
        errors: List of errors encountered (if any)
    """

    results: List[GenerationResult]
    total_tokens: int
    total_latency_ms: int
    successful_count: int
    failed_count: int
    errors: List[str]


class BatchInferenceEngine:
    """
    Engine for batch inference operations.

    This class provides efficient batch processing of multiple prompts,
    with support for parallel execution and error handling.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        max_workers: int = 4
    ):
        """
        Initialize batch inference engine.

        Args:
            inference_engine: InferenceEngine instance
            max_workers: Maximum number of parallel workers
        """
        self.inference_engine = inference_engine
        self.max_workers = max_workers
        logger.info(f"BatchInferenceEngine initialized with {max_workers} workers")

    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        parallel: bool = False
    ) -> BatchGenerationResult:
        """
        Generate completions for multiple prompts.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters (same for all prompts)
            parallel: Whether to process prompts in parallel

        Returns:
            BatchGenerationResult with all results

        Note:
            Parallel processing may not provide benefits with mlx-lm
            depending on the model and hardware configuration.
        """
        if not prompts:
            return BatchGenerationResult(
                results=[],
                total_tokens=0,
                total_latency_ms=0,
                successful_count=0,
                failed_count=0,
                errors=[]
            )

        logger.info(f"Starting batch generation for {len(prompts)} prompts (parallel={parallel})")

        if parallel:
            return self._generate_parallel(prompts, sampling_params)
        else:
            return self._generate_sequential(prompts, sampling_params)

    def _generate_sequential(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams]
    ) -> BatchGenerationResult:
        """
        Generate completions sequentially.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters

        Returns:
            BatchGenerationResult with all results
        """
        results: List[GenerationResult] = []
        errors: List[str] = []
        successful_count = 0
        failed_count = 0
        total_tokens = 0
        total_latency_ms = 0

        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                result = self.inference_engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params
                )
                results.append(result)
                successful_count += 1
                total_tokens += result.tokens_generated
                total_latency_ms += result.latency_ms

            except Exception as e:
                logger.error(f"Failed to generate for prompt {i+1}: {e}")
                failed_count += 1
                errors.append(f"Prompt {i+1}: {str(e)}")

                # Add placeholder result
                results.append(GenerationResult(
                    text="",
                    tokens_generated=0,
                    latency_ms=0,
                    tokens_per_second=0.0,
                    finish_reason="error"
                ))

        logger.info(
            f"Batch generation complete: {successful_count} successful, "
            f"{failed_count} failed, {total_tokens} total tokens"
        )

        return BatchGenerationResult(
            results=results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            successful_count=successful_count,
            failed_count=failed_count,
            errors=errors
        )

    def _generate_parallel(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams]
    ) -> BatchGenerationResult:
        """
        Generate completions in parallel.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters

        Returns:
            BatchGenerationResult with all results

        Note:
            Results will be in the same order as input prompts.
        """
        results: List[Optional[GenerationResult]] = [None] * len(prompts)
        errors: List[str] = []
        successful_count = 0
        failed_count = 0
        total_tokens = 0
        total_latency_ms = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.inference_engine.generate,
                    prompt=prompt,
                    sampling_params=sampling_params
                ): i
                for i, prompt in enumerate(prompts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    result = future.result()
                    results[i] = result
                    successful_count += 1
                    total_tokens += result.tokens_generated
                    total_latency_ms += result.latency_ms

                except Exception as e:
                    logger.error(f"Failed to generate for prompt {i+1}: {e}")
                    failed_count += 1
                    errors.append(f"Prompt {i+1}: {str(e)}")

                    # Add placeholder result
                    results[i] = GenerationResult(
                        text="",
                        tokens_generated=0,
                        latency_ms=0,
                        tokens_per_second=0.0,
                        finish_reason="error"
                    )

        logger.info(
            f"Parallel batch generation complete: {successful_count} successful, "
            f"{failed_count} failed, {total_tokens} total tokens"
        )

        return BatchGenerationResult(
            results=results,  # type: ignore
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            successful_count=successful_count,
            failed_count=failed_count,
            errors=errors
        )

    def generate_batch_stream(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None
    ) -> Iterator[tuple[int, str]]:
        """
        Generate completions for multiple prompts with streaming.

        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters

        Yields:
            Tuples of (prompt_index, token) for each generated token

        Note:
            Prompts are processed sequentially, but each one streams its output.
        """
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Streaming prompt {i+1}/{len(prompts)}")
                for token in self.inference_engine.generate_stream(
                    prompt=prompt,
                    sampling_params=sampling_params
                ):
                    yield (i, token)

            except Exception as e:
                logger.error(f"Failed to stream for prompt {i+1}: {e}")
                # Continue to next prompt on error


def create_batch_request(
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
    **metadata
) -> BatchGenerationRequest:
    """
    Create a batch generation request.

    Args:
        prompts: List of prompts to process
        sampling_params: Sampling parameters for all prompts
        **metadata: Additional metadata to attach

    Returns:
        BatchGenerationRequest instance
    """
    return BatchGenerationRequest(
        prompts=prompts,
        sampling_params=sampling_params,
        metadata=metadata or None
    )
