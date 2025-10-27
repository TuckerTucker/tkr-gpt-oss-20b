"""
Harmony channel extraction utilities.

This module provides utilities for extracting and validating multi-channel
responses from OpenAI Harmony format. Harmony format allows models to output
multiple channels (analysis, commentary, final) within a single response.

Channel Types:
    - analysis: Internal reasoning and chain-of-thought
    - commentary: Meta-commentary about the response
    - final: The actual response content for the user

Format Reference:
    https://github.com/openai/openai-harmony

Examples:
    >>> text = "<|start|>assistant<|channel|>final<|message|>Hello<|end|>"
    >>> extract_channel(text, "final")
    'Hello'

    >>> extract_all_channels(text)
    {'final': 'Hello'}
"""

import re
from typing import Dict, Optional

# Compile regex patterns at module level for performance
# Pattern matches: <|start|>assistant<|channel|>CHANNEL_NAME<|message|>CONTENT<|end|>
_CHANNEL_PATTERN = re.compile(
    r"<\|start\|>assistant<\|channel\|>([^<]+)<\|message\|>(.*?)<\|end\|>",
    re.DOTALL
)

# Pattern to validate basic Harmony format structure
# Matches both simple messages and channel messages
_HARMONY_VALIDATION_PATTERN = re.compile(
    r"<\|start\|>[^<]+(?:<\|channel\|>[^<]+)?<\|message\|>.*?<\|end\|>",
    re.DOTALL
)


def extract_channel(text: str, channel_name: str) -> Optional[str]:
    """
    Extract content from specific channel in Harmony response.

    Searches for channel markers in the Harmony-formatted text and returns
    the content of the specified channel. Returns None if the channel is
    not found.

    Args:
        text: Harmony-formatted response text
        channel_name: Channel to extract (e.g., "analysis", "commentary", "final")

    Returns:
        Channel content as string, or None if channel not found

    Examples:
        >>> response = "<|start|>assistant<|channel|>final<|message|>Hello<|end|>"
        >>> extract_channel(response, "final")
        'Hello'

        >>> extract_channel(response, "analysis")
        None

        >>> # Multi-channel example
        >>> response = '''<|start|>assistant<|channel|>analysis<|message|>Think step by step<|end|>
        ... <|start|>assistant<|channel|>final<|message|>The answer is 42<|end|>'''
        >>> extract_channel(response, "analysis")
        'Think step by step'
        >>> extract_channel(response, "final")
        'The answer is 42'
    """
    if not text:
        return None

    try:
        for match in _CHANNEL_PATTERN.finditer(text):
            channel = match.group(1).strip()
            content = match.group(2).strip()

            if channel == channel_name:
                return content

        return None
    except Exception:
        # Handle any regex errors gracefully
        return None


def extract_all_channels(text: str) -> Dict[str, str]:
    """
    Extract all channels from Harmony response.

    Parses the entire Harmony-formatted response and returns a dictionary
    mapping channel names to their content. If the text is malformed or
    contains no valid channels, returns an empty dictionary.

    Args:
        text: Harmony-formatted response text

    Returns:
        Dictionary mapping channel names to content strings
        Empty dictionary if no channels found or text is malformed

    Examples:
        >>> response = '''<|start|>assistant<|channel|>analysis<|message|>Step 1: Parse input
        ... Step 2: Process data<|end|>
        ... <|start|>assistant<|channel|>final<|message|>Result: Success<|end|>'''
        >>> channels = extract_all_channels(response)
        >>> channels['analysis']
        'Step 1: Parse input\\nStep 2: Process data'
        >>> channels['final']
        'Result: Success'

        >>> # Malformed input
        >>> extract_all_channels("not harmony format")
        {}

        >>> # Empty input
        >>> extract_all_channels("")
        {}
    """
    if not text:
        return {}

    channels = {}

    try:
        for match in _CHANNEL_PATTERN.finditer(text):
            channel = match.group(1).strip()
            content = match.group(2).strip()

            # If duplicate channel, keep the last occurrence
            channels[channel] = content

        return channels
    except Exception:
        # Handle any regex errors gracefully
        return {}


def validate_harmony_format(text: str) -> bool:
    """
    Validate if text follows Harmony format structure.

    Checks if the text contains valid Harmony format markers. Does not
    validate semantic correctness, only structural patterns.

    Args:
        text: Text to validate

    Returns:
        True if text contains valid Harmony format markers, False otherwise

    Examples:
        >>> validate_harmony_format("<|start|>user<|message|>Hello<|end|>")
        True

        >>> validate_harmony_format("<|start|>assistant<|channel|>final<|message|>Hi<|end|>")
        True

        >>> validate_harmony_format("Plain text without markers")
        False

        >>> validate_harmony_format("")
        False

        >>> # Partial markers don't validate
        >>> validate_harmony_format("<|start|>incomplete")
        False
    """
    if not text:
        return False

    try:
        return bool(_HARMONY_VALIDATION_PATTERN.search(text))
    except Exception:
        # Handle any regex errors gracefully
        return False


def format_reasoning_trace(reasoning: str, max_length: int = 500) -> str:
    """
    Format reasoning trace for display with truncation.

    Cleans up reasoning text from the analysis channel by removing Harmony
    markers and truncating if necessary. Preserves line breaks and structure
    while adding a [TRUNCATED] marker if content exceeds max_length.

    Args:
        reasoning: Raw reasoning text from analysis channel
        max_length: Maximum length before truncation (default: 500)

    Returns:
        Formatted reasoning text with [TRUNCATED] marker if truncated

    Examples:
        >>> reasoning = "Step 1: Parse input\\nStep 2: Process"
        >>> format_reasoning_trace(reasoning, max_length=100)
        'Step 1: Parse input\\nStep 2: Process'

        >>> long_text = "A" * 600
        >>> result = format_reasoning_trace(long_text, max_length=100)
        >>> len(result) <= 113  # 100 + len(" [TRUNCATED]")
        True
        >>> result.endswith("[TRUNCATED]")
        True

        >>> # Empty input
        >>> format_reasoning_trace("")
        ''

        >>> # Remove channel markers if present
        >>> with_markers = "<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>"
        >>> format_reasoning_trace(with_markers)
        'Thinking'
    """
    if not reasoning:
        return ""

    try:
        # Remove Harmony markers if present
        cleaned = reasoning

        # Remove channel markers: <|start|>...<|channel|>...<|message|>
        cleaned = re.sub(
            r"<\|start\|>[^<]*<\|channel\|>[^<]*<\|message\|>",
            "",
            cleaned
        )

        # Remove end markers: <|end|>
        cleaned = re.sub(r"<\|end\|>", "", cleaned)

        # Strip leading/trailing whitespace but preserve internal structure
        cleaned = cleaned.strip()

        # Truncate if necessary
        if len(cleaned) > max_length:
            truncated = cleaned[:max_length].rstrip()
            return f"{truncated} [TRUNCATED]"

        return cleaned
    except Exception:
        # Handle any errors gracefully, return original text
        return reasoning[:max_length] if len(reasoning) > max_length else reasoning
