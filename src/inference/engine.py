"""
Inference engine for text generation using mlx-lm.

This module provides the core InferenceEngine class that handles text generation
using Apple's MLX framework for efficient inference on Apple Silicon.
"""

import time
from dataclasses import dataclass
from typing import Optional, Iterator, TYPE_CHECKING, Dict
import logging

from src.inference.exceptions import (
    GenerationError,
    ModelNotLoadedError,
    GenerationCancelledError,
)
from src.inference.metrics import MetricsTracker, GenerationMetrics
from src.sampling.params import SamplingParams
from src.prompts.harmony import HarmonyEncoder, ParsedResponse

if TYPE_CHECKING:
    from src.models.loader import ModelLoader
    from src.config.inference_config import InferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Result of text generation.

    Attributes:
        text: Generated text
        tokens_generated: Number of tokens generated
        latency_ms: Total generation time in milliseconds
        tokens_per_second: Generation throughput
        finish_reason: Reason generation stopped ("stop" | "length" | "error")
        prompt_tokens: Number of tokens in the input prompt (optional)
        metrics: Full metrics object (optional)
    """

    text: str
    tokens_generated: int
    latency_ms: int
    tokens_per_second: float
    finish_reason: str
    prompt_tokens: Optional[int] = None
    metrics: Optional[GenerationMetrics] = None
    reasoning: Optional[str] = None
    commentary: Optional[str] = None
    channels: Optional[Dict[str, str]] = None


class InferenceEngine:
    """
    Text generation engine using mlx-lm.

    This engine provides both synchronous and streaming text generation
    using Apple's MLX framework for efficient inference on Apple Silicon.
    """

    def __init__(self, model_loader: "ModelLoader", config: Optional["InferenceConfig"] = None):
        """
        Initialize inference engine with loaded model and configuration.

        Args:
            model_loader: ModelLoader instance with a loaded model
            config: InferenceConfig with default generation parameters

        Raises:
            ModelNotLoadedError: If model is not loaded
        """
        self.model_loader = model_loader
        self.config = config
        self.metrics_tracker = MetricsTracker()
        self._cancelled = False
        self.harmony_encoder = HarmonyEncoder()

        # Verify model is loaded
        if not self.model_loader.is_loaded():
            raise ModelNotLoadedError("Model must be loaded before creating InferenceEngine")

        logger.info("InferenceEngine initialized successfully")

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> GenerationResult:
        """
        Generate text completion synchronously.

        Args:
            prompt: Input text to complete
            sampling_params: Sampling parameters (uses defaults if not provided)

        Returns:
            GenerationResult with generated text and metadata

        Raises:
            GenerationError: If generation fails
            ModelNotLoadedError: If model is not loaded
        """
        if not self.model_loader.is_loaded():
            raise ModelNotLoadedError()

        # Use default sampling params if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()

        logger.debug(f"Starting generation with prompt length: {len(prompt)}")
        self.metrics_tracker.start_generation()

        try:
            # Import mlx_lm at runtime to avoid import errors if not installed
            try:
                import mlx_lm
            except ImportError as e:
                raise GenerationError(
                    "mlx_lm not installed. Please install with: pip install mlx-lm",
                    details=str(e)
                )

            # Get the model and tokenizer from the loader
            model, tokenizer = self.model_loader.get_model()

            # Prepare generation kwargs from sampling params
            gen_kwargs = self._prepare_generation_kwargs(sampling_params)

            # Generate using mlx_lm.generate()
            start_time = time.time()
            generated_text = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                **gen_kwargs
            )
            end_time = time.time()

            # Calculate metrics
            latency_ms = int((end_time - start_time) * 1000)

            # Estimate token count (rough approximation)
            # In a real implementation, we'd get this from the tokenizer
            tokens_generated = len(generated_text.split())  # Rough estimate

            # Determine finish reason
            finish_reason = "stop"
            if tokens_generated >= sampling_params.max_tokens:
                finish_reason = "length"

            # Record metrics
            prompt_tokens = len(prompt.split())  # Rough estimate
            metrics = self.metrics_tracker.end_generation(
                prompt_tokens=prompt_tokens,
                tokens_generated=tokens_generated,
                finish_reason=finish_reason
            )

            logger.info(
                f"Generation complete: {tokens_generated} tokens in {latency_ms}ms "
                f"({metrics.tokens_per_second:.2f} tokens/sec)"
            )

            # Parse Harmony response if enabled
            if self.config and getattr(self.config, 'use_harmony_format', True):
                parsed = self.harmony_encoder.parse_response(generated_text)
                clean_text = parsed.final

                # Capture reasoning if enabled
                if getattr(self.config, 'capture_reasoning', False):
                    reasoning = parsed.analysis
                    commentary = parsed.commentary
                    channels = {'final': parsed.final}
                    if parsed.analysis:
                        channels['analysis'] = parsed.analysis
                    if parsed.commentary:
                        channels['commentary'] = parsed.commentary
                else:
                    reasoning = None
                    commentary = None
                    channels = None
            else:
                # Legacy mode: use existing extraction
                clean_text = self._extract_final_channel(generated_text)
                reasoning = None
                commentary = None
                channels = None

            return GenerationResult(
                text=clean_text,
                tokens_generated=tokens_generated,
                latency_ms=latency_ms,
                tokens_per_second=metrics.tokens_per_second,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                metrics=metrics,
                reasoning=reasoning,
                commentary=commentary,
                channels=channels,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Try to record metrics even on failure
            try:
                self.metrics_tracker.end_generation(
                    prompt_tokens=len(prompt.split()),
                    tokens_generated=0,
                    finish_reason="error"
                )
            except Exception:
                pass

            raise GenerationError(
                f"Text generation failed: {str(e)}",
                prompt=prompt,
                details=str(e)
            )

    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Iterator[str]:
        """
        Generate text with token streaming.

        Args:
            prompt: Input text to complete
            sampling_params: Sampling parameters (uses defaults if not provided)

        Yields:
            Individual tokens as they're generated

        Raises:
            GenerationError: If generation fails
            ModelNotLoadedError: If model is not loaded
            GenerationCancelledError: If generation is cancelled
        """
        if not self.model_loader.is_loaded():
            raise ModelNotLoadedError()

        # Use default sampling params if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()

        logger.debug(f"Starting streaming generation with prompt length: {len(prompt)}")
        self.metrics_tracker.start_generation()
        self._cancelled = False

        try:
            # Import mlx_lm at runtime
            try:
                import mlx_lm
            except ImportError as e:
                raise GenerationError(
                    "mlx_lm not installed. Please install with: pip install mlx-lm",
                    details=str(e)
                )

            # Get the model and tokenizer from the loader
            model, tokenizer = self.model_loader.get_model()

            # Prepare generation kwargs from sampling params
            gen_kwargs = self._prepare_generation_kwargs(sampling_params)

            # Use mlx_lm.stream_generate() for streaming
            token_count = 0
            first_token = True

            for token in mlx_lm.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                **gen_kwargs
            ):
                # Check for cancellation
                if self._cancelled:
                    logger.info("Generation cancelled by user")
                    self.metrics_tracker.end_generation(
                        prompt_tokens=len(prompt.split()),
                        tokens_generated=token_count,
                        finish_reason="cancelled"
                    )
                    raise GenerationCancelledError()

                # Mark first token for metrics
                if first_token:
                    self.metrics_tracker.mark_first_token()
                    first_token = False

                token_count += 1
                yield token

            # Record final metrics
            finish_reason = "stop"
            if token_count >= sampling_params.max_tokens:
                finish_reason = "length"

            metrics = self.metrics_tracker.end_generation(
                prompt_tokens=len(prompt.split()),
                tokens_generated=token_count,
                finish_reason=finish_reason
            )

            logger.info(
                f"Streaming generation complete: {token_count} tokens "
                f"({metrics.tokens_per_second:.2f} tokens/sec)"
            )

        except GenerationCancelledError:
            raise
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            # Try to record metrics even on failure
            try:
                self.metrics_tracker.end_generation(
                    prompt_tokens=len(prompt.split()),
                    tokens_generated=0,
                    finish_reason="error"
                )
            except Exception:
                pass

            raise GenerationError(
                f"Streaming generation failed: {str(e)}",
                prompt=prompt,
                details=str(e)
            )

    def cancel_generation(self) -> None:
        """
        Cancel ongoing streaming generation.

        Note: This sets a flag that's checked during streaming.
        The cancellation will take effect at the next token boundary.
        """
        self._cancelled = True
        logger.info("Generation cancellation requested")

    def get_reasoning_trace(self, result: GenerationResult) -> Optional[str]:
        """Get reasoning trace from generation result.

        Args:
            result: GenerationResult with potential reasoning

        Returns:
            Reasoning trace or None if not captured
        """
        return result.reasoning

    def get_metrics_summary(self) -> dict:
        """
        Get summary of inference metrics.

        Returns:
            Dictionary containing aggregate metrics
        """
        return self.metrics_tracker.get_summary()

    def reset_metrics(self) -> None:
        """Clear all tracked metrics."""
        self.metrics_tracker.reset()
        logger.info("Metrics tracker reset")

    def _prepare_generation_kwargs(self, sampling_params: SamplingParams) -> dict:
        """
        Convert SamplingParams to mlx_lm generation kwargs.

        Args:
            sampling_params: SamplingParams instance

        Returns:
            Dictionary of kwargs for mlx_lm.generate/stream_generate
        """
        # max_tokens is a direct parameter to stream_generate()
        # Other sampling params go into **kwargs which get passed to the sampler
        kwargs = {
            "max_tokens": sampling_params.max_tokens,
        }

        # Only add sampling parameters if they differ from defaults
        # MLX sampler uses 'temp' not 'temperature'
        if sampling_params.temperature != 1.0:
            kwargs["temp"] = sampling_params.temperature

        if sampling_params.top_p < 1.0:
            kwargs["top_p"] = sampling_params.top_p

        # Note: repetition_penalty might not be supported in all MLX versions
        # Commenting out for now to avoid errors
        # if sampling_params.repetition_penalty != 1.0:
        #     kwargs["repetition_penalty"] = sampling_params.repetition_penalty

        return kwargs

    def _extract_final_channel(self, text: str) -> str:
        """
        Extract the final channel content from Harmony-formatted response.

        GPT-OSS-20B uses the Harmony multi-channel format with three channels:
        - analysis: Internal reasoning (should NOT be shown to users)
        - commentary: Tool/function execution output
        - final: Clean, user-facing response

        Args:
            text: Raw generated text with channel markers

        Returns:
            Extracted final channel content, or original text if no channels found
        """
        # Look for the final channel marker
        final_marker = "<|channel|>final<|message|>"

        if final_marker in text:
            # Extract everything after the final channel marker
            final_start = text.find(final_marker) + len(final_marker)
            final_content = text[final_start:]

            # Clean up any trailing channel markers or end tokens
            final_content = final_content.split("<|end|>")[0]
            final_content = final_content.split("<|start|>")[0]

            return final_content.strip()

        # If no channel markers, check if this is just the analysis channel
        # and try to extract a clean response
        if "<|channel|>analysis" in text:
            # Model may be outputting analysis without final channel
            # In this case, try to extract the last coherent statement
            logger.warning("Model output contains analysis channel but no final channel")

            # Try to find content after <|im_start|>assistant
            if "<|im_start|>assistant" in text:
                content = text.split("<|im_start|>assistant")[1]
                # Remove any channel markers
                content = content.split("<|channel|>")[0]
                content = content.split("<|end|>")[0]
                return content.strip()

        # No channel markers found, return original text
        # This handles legacy responses or non-Harmony formatted output
        return text.strip()
