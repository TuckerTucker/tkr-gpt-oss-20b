"""
Conversation history management.

This module provides the ConversationManager class for managing multi-turn
conversations, including message storage, prompt formatting, and persistence.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from .tokens import count_message_tokens

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """
    Single conversation message.

    Attributes:
        role: Message role (system, user, assistant)
        content: Message content text
        timestamp: ISO format timestamp of message creation
        tokens: Estimated token count for this message
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tokens: int = 0

    def __post_init__(self):
        """Calculate tokens if not provided."""
        if self.tokens == 0:
            self.tokens = count_message_tokens(self.role, self.content)

        # Validate role
        valid_roles = {"system", "user", "assistant"}
        if self.role not in valid_roles:
            logger.warning(
                f"Message role '{self.role}' not in {valid_roles}, proceeding anyway"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(**data)


class ConversationManager:
    """
    Manage conversation history and context.

    This class handles message storage, prompt formatting, token tracking,
    and conversation persistence.

    Attributes:
        messages: List of conversation messages
        max_context_tokens: Maximum context window size
        created_at: Timestamp of conversation creation
        metadata: Optional metadata dictionary
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation manager.

        Args:
            max_context_tokens: Maximum context window size in tokens
            metadata: Optional metadata to store with conversation
        """
        self.messages: List[Message] = []
        self.max_context_tokens = max_context_tokens
        self.created_at = datetime.utcnow().isoformat()
        self.metadata = metadata or {}

        logger.info(
            f"ConversationManager initialized (max_tokens={max_context_tokens})"
        )

    def add_message(self, role: str, content: str) -> Message:
        """
        Add message to conversation history.

        Args:
            role: Message role (system, user, assistant)
            content: Message content

        Returns:
            Created Message object

        Examples:
            >>> conv = ConversationManager()
            >>> msg = conv.add_message("user", "Hello!")
            >>> msg.role
            'user'
            >>> msg.content
            'Hello!'
        """
        message = Message(role=role, content=content)
        self.messages.append(message)

        logger.debug(
            f"Added message #{len(self.messages)} "
            f"(role={role}, tokens={message.tokens})"
        )

        return message

    def get_history(self) -> List[Message]:
        """
        Get all messages in conversation.

        Returns:
            List of Message objects

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hi")
            >>> conv.add_message("assistant", "Hello!")
            >>> len(conv.get_history())
            2
        """
        return self.messages.copy()

    def format_prompt(self, template: str = "chatml") -> str:
        """
        Format conversation into prompt string.

        Args:
            template: Prompt template format (chatml, alpaca, vicuna)

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If template is not supported

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("system", "You are helpful.")
            >>> conv.add_message("user", "Hello!")
            >>> prompt = conv.format_prompt("chatml")
            >>> "<|system|>" in prompt
            True
        """
        # Import here to avoid circular dependency
        from ..prompts.templates import format_conversation

        formatted = format_conversation(self.messages, template)

        logger.debug(
            f"Formatted {len(self.messages)} messages with template '{template}' "
            f"({len(formatted)} chars)"
        )

        return formatted

    def clear(self) -> None:
        """
        Clear conversation history.

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hi")
            >>> conv.clear()
            >>> len(conv.get_history())
            0
        """
        message_count = len(self.messages)
        self.messages.clear()

        logger.info(f"Cleared conversation history ({message_count} messages removed)")

    def get_token_count(self) -> int:
        """
        Get total tokens in conversation.

        Returns:
            Total token count across all messages

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello world")
            >>> conv.get_token_count() > 0
            True
        """
        return sum(msg.tokens for msg in self.messages)

    def get_message_count(self) -> int:
        """
        Get total number of messages.

        Returns:
            Message count
        """
        return len(self.messages)

    def get_last_message(self) -> Optional[Message]:
        """
        Get the most recent message.

        Returns:
            Last message or None if empty
        """
        return self.messages[-1] if self.messages else None

    def get_messages_by_role(self, role: str) -> List[Message]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of messages with matching role
        """
        return [msg for msg in self.messages if msg.role == role]

    def truncate_to_fit(self, reserve_tokens: int = 100) -> int:
        """
        Truncate conversation history to fit within context window.

        Removes oldest messages (except system messages) until conversation
        fits in context window.

        Args:
            reserve_tokens: Tokens to reserve for new content

        Returns:
            Number of messages removed

        Examples:
            >>> conv = ConversationManager(max_context_tokens=100)
            >>> for i in range(20):
            ...     conv.add_message("user", "Message " * 10)
            >>> removed = conv.truncate_to_fit()
            >>> removed > 0
            True
        """
        removed_count = 0
        max_allowed = self.max_context_tokens - reserve_tokens

        # Separate system messages from conversation messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        other_messages = [msg for msg in self.messages if msg.role != "system"]

        # Keep removing oldest non-system messages until we fit
        while other_messages and self.get_token_count() > max_allowed:
            removed_msg = other_messages.pop(0)
            removed_count += 1

            logger.debug(
                f"Truncated message (role={removed_msg.role}, tokens={removed_msg.tokens})"
            )

        # Reconstruct message list with system messages first
        self.messages = system_messages + other_messages

        if removed_count > 0:
            logger.info(
                f"Truncated {removed_count} messages to fit context "
                f"(tokens={self.get_token_count()}/{self.max_context_tokens})"
            )

        return removed_count

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "max_context_tokens": self.max_context_tokens,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "total_tokens": self.get_token_count(),
            "message_count": self.get_message_count(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationManager":
        """
        Create conversation from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ConversationManager instance
        """
        conv = cls(
            max_context_tokens=data.get("max_context_tokens", 4096),
            metadata=data.get("metadata", {})
        )

        # Restore messages
        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            conv.messages.append(msg)

        # Restore timestamp if available
        if "created_at" in data:
            conv.created_at = data["created_at"]

        logger.info(
            f"Loaded conversation from dict ({len(conv.messages)} messages, "
            f"{conv.get_token_count()} tokens)"
        )

        return conv

    def save(self, filepath: str) -> None:
        """
        Save conversation to JSON file.

        Args:
            filepath: Path to save file

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> conv.save("/tmp/test.json")
        """
        data = self.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved conversation to {filepath} "
            f"({len(self.messages)} messages, {self.get_token_count()} tokens)"
        )

    @classmethod
    def load(cls, filepath: str) -> "ConversationManager":
        """
        Load conversation from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            ConversationManager instance

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> conv.save("/tmp/test.json")
            >>> loaded = ConversationManager.load("/tmp/test.json")
            >>> loaded.get_message_count()
            1
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        conv = cls.from_dict(data)

        logger.info(f"Loaded conversation from {filepath}")

        return conv

    def __repr__(self) -> str:
        """String representation of conversation."""
        return (
            f"ConversationManager("
            f"messages={len(self.messages)}, "
            f"tokens={self.get_token_count()}/{self.max_context_tokens})"
        )
