"""
Conversation statistics tracking.

This module provides comprehensive statistics tracking for conversations,
including message counts, token usage, timing metrics, and trend analysis.

Classes:
    ConversationStats: Main statistics tracker for conversations
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .history import ConversationManager, Message

logger = logging.getLogger(__name__)


@dataclass
class MessageStats:
    """
    Statistics for a single message.

    Attributes:
        role: Message role (system, user, assistant)
        tokens: Token count for this message
        timestamp: ISO format timestamp
        content_length: Character length of message content
    """
    role: str
    tokens: int
    timestamp: str
    content_length: int


@dataclass
class ConversationStats:
    """
    Comprehensive statistics tracker for conversations.

    This class tracks various metrics about conversation usage including
    message counts, token usage, timing information, and usage patterns.

    Attributes:
        total_messages: Total number of messages
        total_tokens: Total token count across all messages
        message_breakdown: Message count by role
        token_breakdown: Token count by role
        avg_tokens_per_message: Average tokens per message
        avg_tokens_per_role: Average tokens per message by role
        conversation_duration: Time span from first to last message
        messages_over_time: Chronological message statistics
        created_at: Timestamp when stats were created
    """

    total_messages: int = 0
    total_tokens: int = 0
    message_breakdown: Dict[str, int] = field(default_factory=dict)
    token_breakdown: Dict[str, int] = field(default_factory=dict)
    avg_tokens_per_message: float = 0.0
    avg_tokens_per_role: Dict[str, float] = field(default_factory=dict)
    conversation_duration: Optional[str] = None
    messages_over_time: List[MessageStats] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_conversation(cls, conversation: "ConversationManager") -> "ConversationStats":
        """
        Create statistics from a conversation.

        Args:
            conversation: ConversationManager to analyze

        Returns:
            ConversationStats instance with computed metrics

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello!")
            >>> conv.add_message("assistant", "Hi there!")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> stats.total_messages
            2
        """
        messages = conversation.messages
        total_messages = len(messages)
        total_tokens = sum(msg.tokens for msg in messages)

        # Count messages by role
        message_breakdown = defaultdict(int)
        token_breakdown = defaultdict(int)

        for msg in messages:
            message_breakdown[msg.role] += 1
            token_breakdown[msg.role] += msg.tokens

        # Calculate average tokens per message
        avg_tokens_per_message = total_tokens / max(1, total_messages)

        # Calculate average tokens per role
        avg_tokens_per_role = {}
        for role, count in message_breakdown.items():
            if count > 0:
                avg_tokens_per_role[role] = token_breakdown[role] / count

        # Calculate conversation duration
        conversation_duration = None
        if total_messages > 0:
            try:
                first_time = datetime.fromisoformat(messages[0].timestamp)
                last_time = datetime.fromisoformat(messages[-1].timestamp)
                duration = last_time - first_time
                conversation_duration = str(duration)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to calculate conversation duration: {e}")

        # Build messages over time
        messages_over_time = [
            MessageStats(
                role=msg.role,
                tokens=msg.tokens,
                timestamp=msg.timestamp,
                content_length=len(msg.content)
            )
            for msg in messages
        ]

        logger.info(
            f"Computed conversation statistics: {total_messages} messages, "
            f"{total_tokens} tokens"
        )

        return cls(
            total_messages=total_messages,
            total_tokens=total_tokens,
            message_breakdown=dict(message_breakdown),
            token_breakdown=dict(token_breakdown),
            avg_tokens_per_message=avg_tokens_per_message,
            avg_tokens_per_role=avg_tokens_per_role,
            conversation_duration=conversation_duration,
            messages_over_time=messages_over_time,
        )

    def get_role_percentage(self, role: str) -> float:
        """
        Get percentage of messages for a specific role.

        Args:
            role: Role to calculate percentage for

        Returns:
            Percentage (0-100) of messages from this role

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> for _ in range(3):
            ...     conv.add_message("user", "Test")
            >>> conv.add_message("assistant", "Response")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> stats.get_role_percentage("user")
            75.0
        """
        if self.total_messages == 0:
            return 0.0

        count = self.message_breakdown.get(role, 0)
        return (count / self.total_messages) * 100

    def get_token_percentage(self, role: str) -> float:
        """
        Get percentage of tokens for a specific role.

        Args:
            role: Role to calculate percentage for

        Returns:
            Percentage (0-100) of tokens from this role

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Short")
            >>> conv.add_message("assistant", "Much longer response here")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> user_pct = stats.get_token_percentage("user")
            >>> assistant_pct = stats.get_token_percentage("assistant")
            >>> assistant_pct > user_pct
            True
        """
        if self.total_tokens == 0:
            return 0.0

        count = self.token_breakdown.get(role, 0)
        return (count / self.total_tokens) * 100

    def get_token_efficiency(self) -> float:
        """
        Calculate token efficiency (tokens per character).

        Returns:
            Average tokens per character across all messages

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Test message")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> efficiency = stats.get_token_efficiency()
            >>> 0 < efficiency < 1
            True
        """
        if not self.messages_over_time:
            return 0.0

        total_chars = sum(msg.content_length for msg in self.messages_over_time)
        if total_chars == 0:
            return 0.0

        return self.total_tokens / total_chars

    def get_average_response_time(self) -> Optional[float]:
        """
        Calculate average time between user messages and assistant responses.

        Returns:
            Average response time in seconds, or None if insufficient data

        Examples:
            >>> from src.conversation import ConversationManager
            >>> import time
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Question")
            >>> time.sleep(0.1)
            >>> conv.add_message("assistant", "Answer")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> rt = stats.get_average_response_time()
            >>> rt is None or rt >= 0
            True
        """
        if len(self.messages_over_time) < 2:
            return None

        response_times = []

        for i in range(len(self.messages_over_time) - 1):
            current = self.messages_over_time[i]
            next_msg = self.messages_over_time[i + 1]

            # Check if this is a user -> assistant sequence
            if current.role == "user" and next_msg.role == "assistant":
                try:
                    current_time = datetime.fromisoformat(current.timestamp)
                    next_time = datetime.fromisoformat(next_msg.timestamp)
                    duration = (next_time - current_time).total_seconds()
                    response_times.append(duration)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse timestamps: {e}")
                    continue

        if not response_times:
            return None

        return sum(response_times) / len(response_times)

    def get_token_usage_trend(self) -> List[Dict[str, Any]]:
        """
        Get token usage trend over time.

        Returns:
            List of dictionaries with timestamp and cumulative token count

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "First")
            >>> conv.add_message("user", "Second")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> trend = stats.get_token_usage_trend()
            >>> len(trend) == 2
            True
            >>> trend[1]["cumulative_tokens"] >= trend[0]["cumulative_tokens"]
            True
        """
        trend = []
        cumulative_tokens = 0

        for msg_stat in self.messages_over_time:
            cumulative_tokens += msg_stat.tokens
            trend.append({
                "timestamp": msg_stat.timestamp,
                "role": msg_stat.role,
                "tokens": msg_stat.tokens,
                "cumulative_tokens": cumulative_tokens,
            })

        return trend

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary.

        Returns:
            Dictionary representation of statistics

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Test")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> d = stats.to_dict()
            >>> "total_messages" in d
            True
        """
        return asdict(self)

    def to_summary(self) -> str:
        """
        Generate human-readable summary of statistics.

        Returns:
            Formatted string with key statistics

        Examples:
            >>> from src.conversation import ConversationManager
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> conv.add_message("assistant", "Hi!")
            >>> stats = ConversationStats.from_conversation(conv)
            >>> summary = stats.to_summary()
            >>> "2 messages" in summary
            True
        """
        lines = [
            "=== Conversation Statistics ===",
            f"Total Messages: {self.total_messages}",
            f"Total Tokens: {self.total_tokens}",
            f"Average Tokens/Message: {self.avg_tokens_per_message:.1f}",
            "",
            "Message Breakdown:",
        ]

        for role, count in sorted(self.message_breakdown.items()):
            pct = self.get_role_percentage(role)
            lines.append(f"  {role}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("Token Breakdown:")

        for role, count in sorted(self.token_breakdown.items()):
            pct = self.get_token_percentage(role)
            avg = self.avg_tokens_per_role.get(role, 0)
            lines.append(f"  {role}: {count} tokens ({pct:.1f}%), avg {avg:.1f} per message")

        if self.conversation_duration:
            lines.append("")
            lines.append(f"Conversation Duration: {self.conversation_duration}")

        avg_response = self.get_average_response_time()
        if avg_response is not None:
            lines.append(f"Avg Response Time: {avg_response:.2f}s")

        efficiency = self.get_token_efficiency()
        if efficiency > 0:
            lines.append(f"Token Efficiency: {efficiency:.3f} tokens/char")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of statistics."""
        return (
            f"ConversationStats("
            f"messages={self.total_messages}, "
            f"tokens={self.total_tokens}, "
            f"avg={self.avg_tokens_per_message:.1f})"
        )


def compute_statistics(conversation: "ConversationManager") -> ConversationStats:
    """
    Compute comprehensive statistics for a conversation.

    Convenience function that creates ConversationStats from a conversation.

    Args:
        conversation: ConversationManager to analyze

    Returns:
        ConversationStats instance with computed metrics

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Test")
        >>> stats = compute_statistics(conv)
        >>> stats.total_messages
        1
    """
    return ConversationStats.from_conversation(conversation)


def compare_conversations(
    conv1: "ConversationManager",
    conv2: "ConversationManager"
) -> Dict[str, Any]:
    """
    Compare statistics between two conversations.

    Args:
        conv1: First conversation to compare
        conv2: Second conversation to compare

    Returns:
        Dictionary with comparison metrics

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv1 = ConversationManager()
        >>> conv1.add_message("user", "Short")
        >>> conv2 = ConversationManager()
        >>> for _ in range(5):
        ...     conv2.add_message("user", "Longer message here")
        >>> comparison = compare_conversations(conv1, conv2)
        >>> comparison["message_difference"]
        4
    """
    stats1 = ConversationStats.from_conversation(conv1)
    stats2 = ConversationStats.from_conversation(conv2)

    return {
        "conversation_1": {
            "messages": stats1.total_messages,
            "tokens": stats1.total_tokens,
            "avg_tokens": stats1.avg_tokens_per_message,
        },
        "conversation_2": {
            "messages": stats2.total_messages,
            "tokens": stats2.total_tokens,
            "avg_tokens": stats2.avg_tokens_per_message,
        },
        "differences": {
            "message_difference": stats2.total_messages - stats1.total_messages,
            "token_difference": stats2.total_tokens - stats1.total_tokens,
            "avg_token_difference": stats2.avg_tokens_per_message - stats1.avg_tokens_per_message,
        },
        "ratios": {
            "message_ratio": stats2.total_messages / max(1, stats1.total_messages),
            "token_ratio": stats2.total_tokens / max(1, stats1.total_tokens),
        },
    }
