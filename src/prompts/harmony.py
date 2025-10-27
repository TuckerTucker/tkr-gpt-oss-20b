"""
Harmony Encoder - Multi-channel conversation encoding and parsing.

This module implements the HarmonyEncoder wrapper for OpenAI's harmony package,
providing encoding and multi-channel parsing capabilities.

Integration Contract Compliance:
- Implements HarmonyEncoderProtocol from harmony_encoder_interface.py
- Uses openai-harmony>=0.1.0 with fallback to regex-based parsing
- Thread-safe implementation with no shared mutable state
- Performance targets: <10ms for 20 messages, <5ms for 1KB response parsing
"""

import logging
import re
from typing import List, Optional

# Import dataclasses and enums from the integration contract
import sys
import os

# Add the contract path to allow imports
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-integration', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_encoder_interface import (
    Role,
    Channel,
    HarmonyMessage,
    ParsedResponse,
)

logger = logging.getLogger(__name__)

# Try to import openai-harmony, fall back to regex-based parsing if unavailable
try:
    import openai_harmony
    HARMONY_AVAILABLE = True
    logger.debug("openai-harmony package loaded successfully")
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning(
        "openai-harmony package not available, using fallback regex-based parsing"
    )


class HarmonyEncoder:
    """
    Encoder for multi-channel Harmony format conversations.

    This class wraps the openai-harmony library and provides encoding/parsing
    capabilities with fallback to regex-based parsing when the package is unavailable.

    Thread-safe: All methods are stateless and thread-safe.

    Example:
        >>> encoder = HarmonyEncoder()
        >>> messages = [
        ...     HarmonyMessage(Role.SYSTEM, "You are helpful."),
        ...     HarmonyMessage(Role.USER, "Hello!")
        ... ]
        >>> prompt = encoder.encode_conversation(messages)
        >>> response = "<|start|>assistant<|channel|>final<|message|>Hi!<|end|>"
        >>> parsed = encoder.parse_response(response)
        >>> print(parsed.final)
        'Hi!'
    """

    # Harmony format tokens
    START_TOKEN = "<|start|>"
    END_TOKEN = "<|end|>"
    CHANNEL_TOKEN = "<|channel|>"
    MESSAGE_TOKEN = "<|message|>"

    def __init__(self) -> None:
        """Initialize the HarmonyEncoder."""
        self._harmony_available = HARMONY_AVAILABLE
        logger.debug("HarmonyEncoder initialized (harmony available: %s)", HARMONY_AVAILABLE)

    def encode_conversation(
        self,
        messages: List[HarmonyMessage],
        include_generation_prompt: bool = True
    ) -> str:
        """
        Encode conversation into Harmony format.

        Args:
            messages: List of HarmonyMessage objects
            include_generation_prompt: Add assistant start token

        Returns:
            Harmony-formatted string ready for model input

        Raises:
            ValueError: If message structure is invalid

        Example:
            >>> encoder = HarmonyEncoder()
            >>> messages = [
            ...     HarmonyMessage(Role.SYSTEM, "You are helpful."),
            ...     HarmonyMessage(Role.USER, "Hello!")
            ... ]
            >>> prompt = encoder.encode_conversation(messages)
        """
        logger.debug(
            "Encoding conversation with %d messages (include_generation_prompt=%s)",
            len(messages),
            include_generation_prompt
        )

        # Validate all messages
        for i, msg in enumerate(messages):
            try:
                msg.validate()
            except ValueError as e:
                raise ValueError(f"Invalid message at index {i}: {e}")

        # Build the conversation string
        parts = []

        for msg in messages:
            # Start token with role
            parts.append(f"{self.START_TOKEN}{msg.role.value}")

            # Add channel for assistant messages
            if msg.role == Role.ASSISTANT and msg.channel:
                parts.append(f"{self.CHANNEL_TOKEN}{msg.channel.value}")

            # Add message content
            parts.append(f"{self.MESSAGE_TOKEN}{msg.content}")

            # End token
            parts.append(self.END_TOKEN)

        # Add generation prompt if requested
        if include_generation_prompt:
            parts.append(f"{self.START_TOKEN}assistant")

        result = "".join(parts)
        logger.debug("Encoded conversation to %d characters", len(result))
        return result

    def parse_response(
        self,
        response: str,
        extract_final_only: bool = False
    ) -> ParsedResponse:
        """
        Parse model response into structured channels.

        Args:
            response: Raw model output with channel markers
            extract_final_only: If True, only extract final channel

        Returns:
            ParsedResponse with separated channels

        Raises:
            ValueError: If response format is malformed

        Example:
            >>> encoder = HarmonyEncoder()
            >>> response = "<|start|>assistant<|channel|>final<|message|>Hello!<|end|>"
            >>> parsed = encoder.parse_response(response)
            >>> print(parsed.final)
            'Hello!'
        """
        logger.debug(
            "Parsing response (%d chars, extract_final_only=%s)",
            len(response),
            extract_final_only
        )

        if self._harmony_available:
            return self._parse_with_harmony(response, extract_final_only)
        else:
            return self._parse_with_regex(response, extract_final_only)

    def _parse_with_harmony(
        self,
        response: str,
        extract_final_only: bool
    ) -> ParsedResponse:
        """Parse response using openai-harmony library."""
        logger.debug("Using openai-harmony for parsing")

        # This is a placeholder for actual openai-harmony usage
        # Since the package may not be installed yet, we'll use regex fallback
        # In production, this would use: openai_harmony.parse(response)
        logger.warning("openai-harmony parsing not yet implemented, using regex fallback")
        return self._parse_with_regex(response, extract_final_only)

    def _parse_with_regex(
        self,
        response: str,
        extract_final_only: bool
    ) -> ParsedResponse:
        """Parse response using regex-based fallback parser."""
        logger.debug("Using regex-based fallback parser")

        # Initialize result containers
        final = ""
        analysis = None
        commentary = None

        # Pattern to match assistant messages with channels
        # Format: <|start|>assistant<|channel|>CHANNEL<|message|>CONTENT<|end|>
        pattern = (
            r"<\|start\|>assistant"
            r"<\|channel\|>(analysis|commentary|final)"
            r"<\|message\|>(.*?)"
            r"<\|end\|>"
        )

        matches = re.finditer(pattern, response, re.DOTALL)

        found_any = False
        for match in matches:
            found_any = True
            channel_name = match.group(1)
            content = match.group(2).strip()

            if channel_name == "final":
                final = content
                if extract_final_only:
                    # Early exit if we only need final
                    break
            elif not extract_final_only:
                if channel_name == "analysis":
                    analysis = content
                elif channel_name == "commentary":
                    commentary = content

        # Validate that we found at least the final channel
        if not found_any:
            # Try to extract any text as final fallback
            # Remove harmony tokens and use raw content
            cleaned = re.sub(r'<\|[^|]+\|>', '', response).strip()
            if cleaned:
                logger.warning(
                    "No valid Harmony channels found, using cleaned text as final"
                )
                final = cleaned
            else:
                raise ValueError(
                    "Response does not contain valid Harmony format. "
                    "Expected format: <|start|>assistant<|channel|>final<|message|>CONTENT<|end|>"
                )

        result = ParsedResponse(
            final=final,
            analysis=analysis,
            commentary=commentary,
            raw=response
        )

        logger.debug(
            "Parsed response: final=%d chars, analysis=%s, commentary=%s",
            len(final),
            "present" if analysis else "none",
            "present" if commentary else "none"
        )

        return result

    def validate_format(self, text: str) -> bool:
        """
        Validate if text is properly formatted Harmony.

        Args:
            text: Text to validate

        Returns:
            True if valid Harmony format, False otherwise
        """
        logger.debug("Validating Harmony format for %d character text", len(text))

        # Check for basic Harmony structure
        # Must have balanced start/end tokens
        start_count = text.count(self.START_TOKEN)
        end_count = text.count(self.END_TOKEN)

        if start_count == 0 or end_count == 0:
            logger.debug("Validation failed: no start/end tokens found")
            return False

        if start_count != end_count:
            logger.debug(
                "Validation failed: unbalanced tokens (start=%d, end=%d)",
                start_count,
                end_count
            )
            return False

        # Check for valid role after start token
        valid_roles = [r.value for r in Role]
        pattern = r"<\|start\|>(" + "|".join(valid_roles) + r")"

        if not re.search(pattern, text):
            logger.debug("Validation failed: no valid role found after start token")
            return False

        # Check for proper message structure
        # Each block should be: <|start|>role[<|channel|>channel]<|message|>content<|end|>
        basic_pattern = (
            r"<\|start\|>(" + "|".join(valid_roles) + r")"
            r"(?:<\|channel\|>\w+)?"
            r"<\|message\|>.*?"
            r"<\|end\|>"
        )

        # Find all message blocks
        matches = list(re.finditer(basic_pattern, text, re.DOTALL))

        # Should have at least one complete message block
        if not matches:
            logger.debug("Validation failed: no complete message blocks found")
            return False

        logger.debug("Validation passed: found %d valid message blocks", len(matches))
        return True


# Export the encoder and contract types
__all__ = [
    "HarmonyEncoder",
    "HarmonyMessage",
    "ParsedResponse",
    "Role",
    "Channel",
]
