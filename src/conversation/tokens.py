"""
Token counting utilities for conversation management.

This module provides token counting functionality for managing context windows
and tracking conversation size. Uses simple heuristic estimation (chars/4)
which approximates GPT-style tokenization.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text.

    Uses a simple heuristic: ~4 characters per token on average.
    This is an approximation and may differ from actual tokenizer output.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count

    Examples:
        >>> estimate_tokens("Hello, world!")
        4
        >>> estimate_tokens("This is a longer sentence with more words.")
        11
    """
    if not text:
        return 0

    # Simple heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def count_message_tokens(role: str, content: str) -> int:
    """
    Count tokens in a single message including formatting overhead.

    Accounts for:
    - Role identifier tokens
    - Content tokens
    - Formatting tokens (delimiters, etc.)

    Args:
        role: Message role (system, user, assistant)
        content: Message content

    Returns:
        Total token count for the message
    """
    # Role typically adds 1-2 tokens
    role_tokens = 2

    # Content tokens
    content_tokens = estimate_tokens(content)

    # Formatting overhead (delimiters, newlines, etc.)
    formatting_tokens = 3

    return role_tokens + content_tokens + formatting_tokens


def count_conversation_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Count total tokens in a conversation history.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        Total token count for all messages

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> count_conversation_tokens(messages)
        18
    """
    total = 0

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += count_message_tokens(role, content)

    # Add conversation-level overhead
    conversation_overhead = 3

    return total + conversation_overhead


def estimate_tokens_remaining(
    current_tokens: int,
    max_tokens: int,
    reserve_tokens: int = 100
) -> int:
    """
    Calculate tokens remaining in context window.

    Args:
        current_tokens: Current token count
        max_tokens: Maximum context window size
        reserve_tokens: Tokens to reserve for safety margin

    Returns:
        Number of tokens available for additional content

    Examples:
        >>> estimate_tokens_remaining(1000, 2048, 100)
        948
    """
    available = max_tokens - current_tokens - reserve_tokens
    return max(0, available)


def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    suffix: str = "..."
) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        suffix: Suffix to add if truncated

    Returns:
        Truncated text

    Examples:
        >>> truncate_to_token_limit("This is a long text", 3)
        'This is...'
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    # Calculate approximate character limit
    # Account for suffix tokens
    suffix_tokens = estimate_tokens(suffix)
    available_tokens = max_tokens - suffix_tokens

    # Convert back to approximate characters
    char_limit = available_tokens * 4

    if char_limit <= 0:
        return suffix

    return text[:char_limit].rstrip() + suffix


class TokenCounter:
    """
    Stateful token counter for tracking conversation token usage.

    Attributes:
        total_tokens: Running total of all tokens processed
        max_context_tokens: Maximum context window size
    """

    def __init__(self, max_context_tokens: int = 4096):
        """
        Initialize token counter.

        Args:
            max_context_tokens: Maximum context window size in tokens
        """
        self.total_tokens = 0
        self.max_context_tokens = max_context_tokens
        logger.debug(
            f"TokenCounter initialized with max_context_tokens={max_context_tokens}"
        )

    def add_message(self, role: str, content: str) -> int:
        """
        Add a message and update token count.

        Args:
            role: Message role
            content: Message content

        Returns:
            Tokens in this message
        """
        tokens = count_message_tokens(role, content)
        self.total_tokens += tokens

        logger.debug(
            f"Added message (role={role}, tokens={tokens}, total={self.total_tokens})"
        )

        return tokens

    def get_remaining(self, reserve: int = 100) -> int:
        """
        Get remaining tokens in context window.

        Args:
            reserve: Tokens to reserve as safety margin

        Returns:
            Available tokens
        """
        return estimate_tokens_remaining(
            self.total_tokens,
            self.max_context_tokens,
            reserve
        )

    def is_within_limit(self, additional_tokens: int = 0) -> bool:
        """
        Check if current + additional tokens fit in context window.

        Args:
            additional_tokens: Additional tokens to check

        Returns:
            True if within limit, False otherwise
        """
        return (self.total_tokens + additional_tokens) <= self.max_context_tokens

    def reset(self) -> None:
        """Reset token counter to zero."""
        logger.debug(f"Resetting token counter (was {self.total_tokens})")
        self.total_tokens = 0

    def get_usage_percentage(self) -> float:
        """
        Get context window usage as percentage.

        Returns:
            Usage percentage (0.0 to 100.0)
        """
        if self.max_context_tokens == 0:
            return 0.0
        return (self.total_tokens / self.max_context_tokens) * 100.0
