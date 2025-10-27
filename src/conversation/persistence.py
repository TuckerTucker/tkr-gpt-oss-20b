"""
Conversation persistence utilities.

This module provides utilities for saving and loading conversations
in various formats, managing conversation archives, and exporting
conversation data.
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .history import ConversationManager, Message

logger = logging.getLogger(__name__)


class ConversationPersistence:
    """
    Handle conversation persistence operations.

    This class provides utilities for saving, loading, and managing
    conversation files.

    Attributes:
        storage_dir: Directory for storing conversation files
    """

    def __init__(self, storage_dir: str = ".chat_history"):
        """
        Initialize conversation persistence.

        Args:
            storage_dir: Directory to store conversation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        logger.info(f"ConversationPersistence initialized (dir={self.storage_dir})")

    def save_conversation(
        self,
        conversation: ConversationManager,
        filename: Optional[str] = None
    ) -> str:
        """
        Save conversation to file.

        Args:
            conversation: ConversationManager to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> persistence = ConversationPersistence()
            >>> path = persistence.save_conversation(conv)
            >>> Path(path).exists()
            True
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        filepath = self.storage_dir / filename
        conversation.save(str(filepath))

        logger.info(f"Saved conversation to {filepath}")
        return str(filepath)

    def load_conversation(self, filename: str) -> ConversationManager:
        """
        Load conversation from file.

        Args:
            filename: Filename to load (without path)

        Returns:
            Loaded ConversationManager

        Examples:
            >>> persistence = ConversationPersistence()
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Test")
            >>> path = persistence.save_conversation(conv, "test.json")
            >>> loaded = persistence.load_conversation("test.json")
            >>> loaded.get_message_count()
            1
        """
        filepath = self.storage_dir / filename
        conversation = ConversationManager.load(str(filepath))

        logger.info(f"Loaded conversation from {filepath}")
        return conversation

    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all saved conversations.

        Returns:
            List of conversation metadata dictionaries

        Examples:
            >>> persistence = ConversationPersistence()
            >>> conversations = persistence.list_conversations()
            >>> isinstance(conversations, list)
            True
        """
        conversations = []

        for filepath in sorted(self.storage_dir.glob("*.json")):
            try:
                # Load metadata without loading full conversation
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                metadata = {
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "created_at": data.get("created_at", "unknown"),
                    "message_count": data.get("message_count", 0),
                    "total_tokens": data.get("total_tokens", 0),
                    "max_context_tokens": data.get("max_context_tokens", 4096),
                    "size_bytes": filepath.stat().st_size,
                }

                # Add custom metadata if present
                if "metadata" in data:
                    metadata["custom_metadata"] = data["metadata"]

                conversations.append(metadata)

            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

        logger.debug(f"Found {len(conversations)} conversations")
        return conversations

    def delete_conversation(self, filename: str) -> bool:
        """
        Delete a saved conversation.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted, False if not found

        Examples:
            >>> persistence = ConversationPersistence()
            >>> conv = ConversationManager()
            >>> path = persistence.save_conversation(conv, "delete_me.json")
            >>> persistence.delete_conversation("delete_me.json")
            True
        """
        filepath = self.storage_dir / filename

        if not filepath.exists():
            logger.warning(f"Conversation file not found: {filepath}")
            return False

        filepath.unlink()
        logger.info(f"Deleted conversation: {filepath}")
        return True

    def export_to_text(
        self,
        conversation: ConversationManager,
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """
        Export conversation to plain text format.

        Args:
            conversation: ConversationManager to export
            filepath: Path to output file
            include_metadata: Whether to include metadata header

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> conv.add_message("assistant", "Hi there!")
            >>> persistence = ConversationPersistence()
            >>> persistence.export_to_text(conv, "/tmp/test.txt")
        """
        lines = []

        if include_metadata:
            lines.append("=" * 60)
            lines.append("CONVERSATION EXPORT")
            lines.append("=" * 60)
            lines.append(f"Created: {conversation.created_at}")
            lines.append(f"Messages: {conversation.get_message_count()}")
            lines.append(f"Tokens: {conversation.get_token_count()}")
            lines.append("=" * 60)
            lines.append("")

        for i, msg in enumerate(conversation.messages, 1):
            lines.append(f"[{i}] {msg.role.upper()}")
            lines.append(f"Timestamp: {msg.timestamp}")
            lines.append(f"Tokens: {msg.tokens}")
            lines.append("-" * 60)
            lines.append(msg.content)
            lines.append("")
            lines.append("=" * 60)
            lines.append("")

        output = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)

        logger.info(f"Exported conversation to text: {filepath}")

    def export_to_markdown(
        self,
        conversation: ConversationManager,
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """
        Export conversation to Markdown format.

        Args:
            conversation: ConversationManager to export
            filepath: Path to output file
            include_metadata: Whether to include metadata header

        Examples:
            >>> conv = ConversationManager()
            >>> conv.add_message("user", "Hello")
            >>> persistence = ConversationPersistence()
            >>> persistence.export_to_markdown(conv, "/tmp/test.md")
        """
        lines = []

        if include_metadata:
            lines.append("# Conversation Export")
            lines.append("")
            lines.append(f"- **Created:** {conversation.created_at}")
            lines.append(f"- **Messages:** {conversation.get_message_count()}")
            lines.append(f"- **Total Tokens:** {conversation.get_token_count()}")
            lines.append("")
            lines.append("---")
            lines.append("")

        for msg in conversation.messages:
            role_emoji = {
                "system": "⚙️",
                "user": "👤",
                "assistant": "🤖"
            }.get(msg.role, "❓")

            lines.append(f"## {role_emoji} {msg.role.upper()}")
            lines.append("")
            lines.append(f"*{msg.timestamp}* | *{msg.tokens} tokens*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        output = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)

        logger.info(f"Exported conversation to markdown: {filepath}")

    def cleanup_old_conversations(self, keep_count: int = 10) -> int:
        """
        Remove old conversation files, keeping most recent N.

        Args:
            keep_count: Number of recent conversations to keep

        Returns:
            Number of files deleted

        Examples:
            >>> persistence = ConversationPersistence()
            >>> # Create many conversations
            >>> for i in range(15):
            ...     conv = ConversationManager()
            ...     persistence.save_conversation(conv)
            >>> deleted = persistence.cleanup_old_conversations(keep_count=10)
            >>> deleted >= 5
            True
        """
        conversations = self.list_conversations()

        if len(conversations) <= keep_count:
            logger.info(f"No cleanup needed ({len(conversations)} <= {keep_count})")
            return 0

        # Sort by creation time (oldest first)
        conversations.sort(key=lambda c: c.get("created_at", ""))

        # Delete oldest conversations
        to_delete = conversations[:-keep_count]
        deleted_count = 0

        for conv_info in to_delete:
            if self.delete_conversation(conv_info["filename"]):
                deleted_count += 1

        logger.info(
            f"Cleaned up {deleted_count} old conversations "
            f"(kept {keep_count} most recent)"
        )

        return deleted_count

    def get_storage_size(self) -> int:
        """
        Get total size of conversation storage in bytes.

        Returns:
            Total size in bytes
        """
        total_size = 0

        for filepath in self.storage_dir.glob("*.json"):
            total_size += filepath.stat().st_size

        return total_size

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.

        Returns:
            Dictionary with storage stats

        Examples:
            >>> persistence = ConversationPersistence()
            >>> stats = persistence.get_storage_stats()
            >>> "total_conversations" in stats
            True
        """
        conversations = self.list_conversations()

        return {
            "storage_dir": str(self.storage_dir),
            "total_conversations": len(conversations),
            "total_size_bytes": self.get_storage_size(),
            "total_messages": sum(c.get("message_count", 0) for c in conversations),
            "total_tokens": sum(c.get("total_tokens", 0) for c in conversations),
        }

    def __repr__(self) -> str:
        """String representation of persistence handler."""
        return f"ConversationPersistence(storage_dir={self.storage_dir})"
