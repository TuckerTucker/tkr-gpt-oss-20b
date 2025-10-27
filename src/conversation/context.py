"""
Context window tracking and management.

This module provides utilities for tracking and managing context window usage,
including automatic truncation strategies and context window optimization.
"""

import logging
from typing import List, Optional, Tuple
from enum import Enum

from .history import Message, ConversationManager
from .tokens import estimate_tokens_remaining

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """
    Strategies for handling context window overflow.

    REMOVE_OLDEST: Remove oldest non-system messages first
    REMOVE_MIDDLE: Keep system, first, and last messages, remove middle
    SLIDING_WINDOW: Keep most recent N messages that fit
    """

    REMOVE_OLDEST = "remove_oldest"
    REMOVE_MIDDLE = "remove_middle"
    SLIDING_WINDOW = "sliding_window"


class ContextWindowTracker:
    """
    Track and manage context window usage.

    This class monitors token usage and provides automatic truncation
    when the context window is full.

    Attributes:
        max_tokens: Maximum context window size
        reserve_tokens: Tokens reserved for response generation
        strategy: Truncation strategy to use
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 100,
        strategy: TruncationStrategy = TruncationStrategy.REMOVE_OLDEST
    ):
        """
        Initialize context window tracker.

        Args:
            max_tokens: Maximum context window size
            reserve_tokens: Tokens to reserve for response
            strategy: Truncation strategy to use
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.strategy = strategy

        logger.info(
            f"ContextWindowTracker initialized "
            f"(max_tokens={max_tokens}, reserve={reserve_tokens}, "
            f"strategy={strategy.value})"
        )

    def get_usage(self, conversation: ConversationManager) -> Tuple[int, int, float]:
        """
        Get current context window usage.

        Args:
            conversation: ConversationManager to check

        Returns:
            Tuple of (used_tokens, available_tokens, usage_percentage)

        Examples:
            >>> conv = ConversationManager(max_context_tokens=2048)
            >>> conv.add_message("user", "Hello")
            >>> tracker = ContextWindowTracker(max_tokens=2048)
            >>> used, avail, pct = tracker.get_usage(conv)
            >>> used > 0
            True
        """
        used_tokens = conversation.get_token_count()
        available_tokens = estimate_tokens_remaining(
            used_tokens,
            self.max_tokens,
            self.reserve_tokens
        )
        usage_pct = (used_tokens / self.max_tokens) * 100.0

        return used_tokens, available_tokens, usage_pct

    def is_overflow(self, conversation: ConversationManager) -> bool:
        """
        Check if context window is overflowing.

        Args:
            conversation: ConversationManager to check

        Returns:
            True if overflow, False otherwise
        """
        used_tokens = conversation.get_token_count()
        max_allowed = self.max_tokens - self.reserve_tokens

        return used_tokens > max_allowed

    def will_overflow(
        self,
        conversation: ConversationManager,
        additional_tokens: int
    ) -> bool:
        """
        Check if adding tokens would cause overflow.

        Args:
            conversation: ConversationManager to check
            additional_tokens: Tokens to potentially add

        Returns:
            True if would overflow, False otherwise
        """
        used_tokens = conversation.get_token_count()
        total_tokens = used_tokens + additional_tokens
        max_allowed = self.max_tokens - self.reserve_tokens

        return total_tokens > max_allowed

    def truncate_if_needed(
        self,
        conversation: ConversationManager
    ) -> int:
        """
        Truncate conversation if context window is full.

        Args:
            conversation: ConversationManager to potentially truncate

        Returns:
            Number of messages removed

        Examples:
            >>> conv = ConversationManager(max_context_tokens=100)
            >>> for i in range(10):
            ...     conv.add_message("user", "Long message " * 20)
            >>> tracker = ContextWindowTracker(max_tokens=100)
            >>> removed = tracker.truncate_if_needed(conv)
            >>> removed > 0
            True
        """
        if not self.is_overflow(conversation):
            logger.debug("No truncation needed")
            return 0

        logger.info(
            f"Context window overflow detected, applying {self.strategy.value} strategy"
        )

        if self.strategy == TruncationStrategy.REMOVE_OLDEST:
            return self._truncate_remove_oldest(conversation)
        elif self.strategy == TruncationStrategy.REMOVE_MIDDLE:
            return self._truncate_remove_middle(conversation)
        elif self.strategy == TruncationStrategy.SLIDING_WINDOW:
            return self._truncate_sliding_window(conversation)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using REMOVE_OLDEST")
            return self._truncate_remove_oldest(conversation)

    def _truncate_remove_oldest(self, conversation: ConversationManager) -> int:
        """
        Remove oldest non-system messages until conversation fits.

        Args:
            conversation: ConversationManager to truncate

        Returns:
            Number of messages removed
        """
        return conversation.truncate_to_fit(self.reserve_tokens)

    def _truncate_remove_middle(self, conversation: ConversationManager) -> int:
        """
        Keep system, first few, and last few messages; remove middle.

        Args:
            conversation: ConversationManager to truncate

        Returns:
            Number of messages removed
        """
        messages = conversation.messages
        max_allowed = self.max_tokens - self.reserve_tokens

        # Separate system messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        if len(other_msgs) <= 4:
            # Too few to use this strategy, fall back to remove_oldest
            return self._truncate_remove_oldest(conversation)

        # Try to keep first 2 and last 2 non-system messages
        keep_start = 2
        keep_end = 2

        removed_count = 0

        # Keep removing from middle until we fit
        while other_msgs and conversation.get_token_count() > max_allowed:
            if len(other_msgs) <= (keep_start + keep_end):
                # Not enough messages, remove from start
                other_msgs.pop(0)
            else:
                # Remove from middle
                middle_idx = keep_start
                other_msgs.pop(middle_idx)

            removed_count += 1

        # Reconstruct conversation
        conversation.messages = system_msgs + other_msgs

        logger.info(
            f"Removed {removed_count} middle messages "
            f"(tokens={conversation.get_token_count()}/{self.max_tokens})"
        )

        return removed_count

    def _truncate_sliding_window(self, conversation: ConversationManager) -> int:
        """
        Keep most recent messages that fit in window.

        Args:
            conversation: ConversationManager to truncate

        Returns:
            Number of messages removed
        """
        messages = conversation.messages
        max_allowed = self.max_tokens - self.reserve_tokens

        # Always keep system messages
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]

        # Calculate tokens used by system messages
        system_tokens = sum(m.tokens for m in system_msgs)
        available_for_others = max_allowed - system_tokens

        # Take messages from end until we run out of space
        kept_msgs: List[Message] = []
        current_tokens = 0

        for msg in reversed(other_msgs):
            if current_tokens + msg.tokens <= available_for_others:
                kept_msgs.insert(0, msg)
                current_tokens += msg.tokens
            else:
                break

        removed_count = len(other_msgs) - len(kept_msgs)

        # Reconstruct conversation
        conversation.messages = system_msgs + kept_msgs

        logger.info(
            f"Sliding window truncation removed {removed_count} messages "
            f"(tokens={conversation.get_token_count()}/{self.max_tokens})"
        )

        return removed_count

    def estimate_messages_remaining(
        self,
        conversation: ConversationManager,
        avg_message_tokens: int = 50
    ) -> int:
        """
        Estimate how many more messages can fit.

        Args:
            conversation: ConversationManager to check
            avg_message_tokens: Average tokens per message

        Returns:
            Estimated number of messages that can be added
        """
        used_tokens = conversation.get_token_count()
        available = estimate_tokens_remaining(
            used_tokens,
            self.max_tokens,
            self.reserve_tokens
        )

        return max(0, available // avg_message_tokens)

    def get_context_summary(
        self,
        conversation: ConversationManager
    ) -> dict:
        """
        Get comprehensive context window summary.

        Args:
            conversation: ConversationManager to analyze

        Returns:
            Dictionary with usage statistics

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> tracker = ContextWindowTracker()
            >>> summary = tracker.get_context_summary(conv)
            >>> "used_tokens" in summary
            True
        """
        used_tokens, available_tokens, usage_pct = self.get_usage(conversation)

        return {
            "used_tokens": used_tokens,
            "available_tokens": available_tokens,
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "usage_percentage": usage_pct,
            "is_overflow": self.is_overflow(conversation),
            "message_count": conversation.get_message_count(),
            "strategy": self.strategy.value,
        }

    def __repr__(self) -> str:
        """String representation of tracker."""
        return (
            f"ContextWindowTracker("
            f"max_tokens={self.max_tokens}, "
            f"reserve={self.reserve_tokens}, "
            f"strategy={self.strategy.value})"
        )
