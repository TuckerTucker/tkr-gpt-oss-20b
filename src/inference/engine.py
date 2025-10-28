"""
Inference engine for text generation using mlx-lm.

This module provides the core InferenceEngine class that handles text generation
using Apple's MLX framework for efficient inference on Apple Silicon.
"""

import time
from dataclasses import dataclass
from typing import Optional, Iterator, TYPE_CHECKING, Dict, Any
import logging

from src.inference.exceptions import (
    GenerationError,
    ModelNotLoadedError,
    GenerationCancelledError,
)
from src.inference.metrics import MetricsTracker, GenerationMetrics
from src.sampling.params import SamplingParams
from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    ReasoningLevel as HarmonyReasoningLevel,
    HarmonyResponseParser,
    ParsedHarmonyResponse
)

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

        # Initialize Harmony prompt builder for token-based prompt creation
        self.harmony_builder = HarmonyPromptBuilder()

        # Initialize Harmony response parser for channel extraction
        self.harmony_parser = HarmonyResponseParser()

        # Verify model is loaded
        if not self.model_loader.is_loaded():
            raise ModelNotLoadedError("Model must be loaded before creating InferenceEngine")

        logger.info("InferenceEngine initialized successfully")

    def _convert_reasoning_level(self, config_level) -> HarmonyReasoningLevel:
        """
        Convert InferenceConfig.ReasoningLevel to Harmony ReasoningLevel.

        Args:
            config_level: ReasoningLevel from InferenceConfig

        Returns:
            HarmonyReasoningLevel for use with HarmonyPromptBuilder
        """
        # Import here to avoid circular imports
        from src.config.inference_config import ReasoningLevel as ConfigReasoningLevel

        if config_level == ConfigReasoningLevel.LOW:
            return HarmonyReasoningLevel.LOW
        elif config_level == ConfigReasoningLevel.HIGH:
            return HarmonyReasoningLevel.HIGH
        else:  # MEDIUM or default
            return HarmonyReasoningLevel.MEDIUM

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> GenerationResult:
        """
        Generate text completion synchronously using Harmony format.

        This method builds a Harmony-compliant prompt with system message,
        developer instructions, and user input, then generates a response
        using MLX.

        Args:
            prompt: User input text (will be wrapped in Harmony format)
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

            # === BUILD HARMONY PROMPT ===

            # Get configuration values with defaults
            from datetime import datetime

            # Determine reasoning level
            if self.config and hasattr(self.config, 'reasoning_level'):
                harmony_reasoning = self._convert_reasoning_level(self.config.reasoning_level)
            else:
                harmony_reasoning = HarmonyReasoningLevel.MEDIUM

            # Get knowledge cutoff (use default if not in config)
            knowledge_cutoff = "2024-06"  # Default until Agent 2A adds it to config
            if self.config and hasattr(self.config, 'knowledge_cutoff'):
                knowledge_cutoff = self.config.knowledge_cutoff

            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            if self.config and hasattr(self.config, 'get_current_date'):
                date_from_config = self.config.get_current_date()
                if date_from_config:
                    current_date = date_from_config

            # Build system prompt with Harmony
            system_prompt = self.harmony_builder.build_system_prompt(
                reasoning_level=harmony_reasoning,
                knowledge_cutoff=knowledge_cutoff,
                current_date=current_date,
                include_function_tools=False  # TODO: Add function tool support later
            )

            # Build developer prompt with default instructions
            developer_prompt = self.harmony_builder.build_developer_prompt(
                instructions="You are a helpful, harmless, and honest AI assistant."
            )

            # Build conversation with user prompt
            messages = [{"role": "user", "content": prompt}]
            conversation_prompt = self.harmony_builder.build_conversation(
                messages=messages,
                system_prompt=system_prompt,
                developer_prompt=developer_prompt,
                include_generation_prompt=True
            )

            # Use token IDs for MLX (MLX accepts List[int])
            prompt_tokens = conversation_prompt.token_ids

            logger.debug(f"Built Harmony prompt with {len(prompt_tokens)} tokens")

            # === END HARMONY PROMPT BUILDING ===

            # Prepare generation kwargs from sampling params
            gen_kwargs = self._prepare_generation_kwargs(sampling_params)

            # Generate using mlx_lm.generate() with token IDs
            start_time = time.time()
            generated_text = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,  # NOW USING TOKEN IDS
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
            # Use actual token count from Harmony prompt instead of rough estimate
            actual_prompt_tokens = len(prompt_tokens)
            metrics = self.metrics_tracker.end_generation(
                prompt_tokens=actual_prompt_tokens,
                tokens_generated=tokens_generated,
                finish_reason=finish_reason
            )

            logger.info(
                f"Generation complete: {tokens_generated} tokens in {latency_ms}ms "
                f"({metrics.tokens_per_second:.2f} tokens/sec)"
            )

            # --- RESPONSE PARSING SECTION BELOW (Agent 2C) ---

            # === NEW: PARSE HARMONY RESPONSE ===

            # Use HarmonyResponseParser to extract channels
            if self.config and getattr(self.config, 'use_harmony_format', True):
                try:
                    # Parse response with Harmony parser
                    extract_final_only = not getattr(self.config, 'capture_reasoning', False)
                    parsed = self.harmony_parser.parse_response_text(
                        response_text=generated_text,
                        tokenizer=tokenizer,
                        extract_final_only=extract_final_only
                    )

                    # Extract final channel for user-facing text
                    clean_text = parsed.final

                    # Capture reasoning if enabled
                    reasoning = parsed.analysis if self.config.capture_reasoning else None
                    commentary = parsed.commentary if self.config.capture_reasoning else None
                    channels = parsed.channels if self.config.capture_reasoning else None

                    logger.debug(
                        f"Parsed Harmony response: final={len(clean_text)} chars, "
                        f"analysis={'yes' if reasoning else 'no'}, "
                        f"commentary={'yes' if commentary else 'no'}"
                    )

                except Exception as e:
                    logger.warning(f"Harmony parsing failed, using raw text: {e}")
                    # Fallback to raw text if parsing fails
                    clean_text = generated_text
                    reasoning = None
                    commentary = None
                    channels = None
            else:
                # Legacy mode: use raw text
                clean_text = generated_text
                reasoning = None
                commentary = None
                channels = None

            # === END NEW SECTION ===

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
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate text with token streaming using Harmony format.

        This method builds a Harmony-compliant prompt with system message,
        developer instructions, and user input, then generates a response
        using MLX with token-by-token streaming and real-time channel detection.

        Args:
            prompt: User input text (will be wrapped in Harmony format)
            sampling_params: Sampling parameters (uses defaults if not provided)

        Yields:
            Dict with keys:
                - token: Generated token (str)
                - channel: Current channel name (str, e.g., "analysis", "final")
                - content: Accumulated content in current channel (str)
                - delta: Content added by this token (str)
                - is_final: Whether this is the final channel (bool)

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

            # === BUILD HARMONY PROMPT (same as non-streaming) ===

            # Get configuration values with defaults
            from datetime import datetime

            # Determine reasoning level
            if self.config and hasattr(self.config, 'reasoning_level'):
                harmony_reasoning = self._convert_reasoning_level(self.config.reasoning_level)
            else:
                harmony_reasoning = HarmonyReasoningLevel.MEDIUM

            # Get knowledge cutoff (use default if not in config)
            knowledge_cutoff = "2024-06"  # Default until Agent 2A adds it to config
            if self.config and hasattr(self.config, 'knowledge_cutoff'):
                knowledge_cutoff = self.config.knowledge_cutoff

            # Get current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            if self.config and hasattr(self.config, 'get_current_date'):
                date_from_config = self.config.get_current_date()
                if date_from_config:
                    current_date = date_from_config

            # Build system prompt with Harmony
            system_prompt = self.harmony_builder.build_system_prompt(
                reasoning_level=harmony_reasoning,
                knowledge_cutoff=knowledge_cutoff,
                current_date=current_date,
                include_function_tools=False  # TODO: Add function tool support later
            )

            # Build developer prompt with default instructions
            developer_prompt = self.harmony_builder.build_developer_prompt(
                instructions="You are a helpful, harmless, and honest AI assistant."
            )

            # Build conversation with user prompt
            messages = [{"role": "user", "content": prompt}]
            conversation_prompt = self.harmony_builder.build_conversation(
                messages=messages,
                system_prompt=system_prompt,
                developer_prompt=developer_prompt,
                include_generation_prompt=True
            )

            # Use token IDs for MLX (MLX accepts List[int])
            prompt_tokens = conversation_prompt.token_ids

            logger.debug(f"Built Harmony prompt with {len(prompt_tokens)} tokens")

            # === END HARMONY PROMPT BUILDING ===

            # === INITIALIZE STREAMABLE PARSER FOR REAL-TIME CHANNEL DETECTION ===

            from openai_harmony import StreamableParser, Role

            # Create parser for streaming channel detection
            # Use ASSISTANT role since we're parsing assistant responses
            parser = StreamableParser(self.harmony_parser._encoding, Role.ASSISTANT)

            logger.debug("StreamableParser initialized for real-time channel detection")

            # === END PARSER INITIALIZATION ===

            # Prepare generation kwargs from sampling params
            gen_kwargs = self._prepare_generation_kwargs(sampling_params)

            # Use mlx_lm.stream_generate() for streaming with token IDs
            token_count = 0
            first_token = True

            for token in mlx_lm.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_tokens,  # NOW USING TOKEN IDS (like non-streaming)
                **gen_kwargs
            ):
                # Check for cancellation
                if self._cancelled:
                    logger.info("Generation cancelled by user")
                    self.metrics_tracker.end_generation(
                        prompt_tokens=len(prompt_tokens),
                        tokens_generated=token_count,
                        finish_reason="cancelled"
                    )
                    raise GenerationCancelledError()

                # Mark first token for metrics
                if first_token:
                    self.metrics_tracker.mark_first_token()
                    first_token = False

                # === REAL-TIME CHANNEL DETECTION ===

                # MLX stream_generate yields strings, but we need token IDs for parser
                # Encode the token string to get token ID
                try:
                    # Encode token to get token ID for parser
                    token_ids = tokenizer.encode(token, allowed_special="all")
                    if token_ids:
                        token_id = token_ids[0]

                        # Process token through StreamableParser for channel detection
                        parser = parser.process(token_id)

                        # Extract current channel and content
                        current_channel = parser.current_channel if parser.current_channel else "unknown"
                        current_content = parser.current_content if parser.current_content else ""
                        last_delta = parser.last_content_delta if parser.last_content_delta else token

                        # Yield token with channel metadata
                        yield {
                            'token': token,
                            'channel': current_channel,
                            'content': current_content,
                            'delta': last_delta,
                            'is_final': current_channel == 'final'
                        }

                    else:
                        # Fallback if encoding fails - yield without channel info
                        logger.warning(f"Failed to encode token for parsing: {token}")
                        yield {
                            'token': token,
                            'channel': 'unknown',
                            'content': token,
                            'delta': token,
                            'is_final': False
                        }

                except Exception as e:
                    # Fallback on parsing errors - continue streaming
                    logger.debug(f"Channel detection error for token: {e}")
                    yield {
                        'token': token,
                        'channel': 'unknown',
                        'content': token,
                        'delta': token,
                        'is_final': False
                    }

                # === END CHANNEL DETECTION ===

                token_count += 1

            # Finalize parser to ensure all channels are processed
            try:
                parser = parser.process_eos()
                logger.debug(f"Finalized StreamableParser with {len(parser.messages)} messages")
            except Exception as e:
                logger.warning(f"Error finalizing parser: {e}")

            # Record final metrics
            finish_reason = "stop"
            if token_count >= sampling_params.max_tokens:
                finish_reason = "length"

            metrics = self.metrics_tracker.end_generation(
                prompt_tokens=len(prompt_tokens),
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

        # NOTE: This version of mlx_lm doesn't accept sampling parameters
        # All sampling params commented out to avoid errors
        # The model will use its default sampling behavior

        # if sampling_params.temperature != 1.0:
        #     kwargs["temp"] = sampling_params.temperature

        # if sampling_params.top_p < 1.0:
        #     kwargs["top_p"] = sampling_params.top_p

        # if sampling_params.repetition_penalty != 1.0:
        #     kwargs["repetition_penalty"] = sampling_params.repetition_penalty

        return kwargs
