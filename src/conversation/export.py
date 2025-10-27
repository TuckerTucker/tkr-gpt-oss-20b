"""
Conversation export functionality.

This module provides utilities for exporting conversations to various formats
including markdown, JSON, and plain text. Each export format includes
comprehensive metadata about the conversation.

Functions:
    export_to_markdown: Export conversation as markdown with formatting
    export_to_json: Export conversation as structured JSON
    export_to_text: Export conversation as plain text
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .history import ConversationManager

logger = logging.getLogger(__name__)


def export_to_markdown(
    conversation: "ConversationManager",
    filepath: str,
    include_metadata: bool = True,
    include_stats: bool = True,
) -> None:
    """
    Export conversation to Markdown format.

    Creates a well-formatted markdown document with optional metadata header,
    conversation statistics, and formatted messages with role-specific emojis.

    Args:
        conversation: ConversationManager to export
        filepath: Path to output file
        include_metadata: Whether to include metadata header (default: True)
        include_stats: Whether to include conversation statistics (default: True)

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Hello!")
        >>> conv.add_message("assistant", "Hi there!")
        >>> export_to_markdown(conv, "/tmp/chat.md")
        >>> Path("/tmp/chat.md").exists()
        True

    Notes:
        - System messages use ⚙️ emoji
        - User messages use 👤 emoji
        - Assistant messages use 🤖 emoji
        - Each message includes timestamp and token count
    """
    lines = []

    if include_metadata:
        lines.append("# Conversation Export")
        lines.append("")
        lines.append(f"**Created:** {conversation.created_at}")
        lines.append(f"**Messages:** {conversation.get_message_count()}")
        lines.append(f"**Total Tokens:** {conversation.get_token_count()}")
        lines.append(f"**Max Context:** {conversation.max_context_tokens}")
        lines.append("")

        # Add custom metadata if present
        if conversation.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in conversation.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        if include_stats:
            # Calculate statistics
            user_msgs = [m for m in conversation.messages if m.role == "user"]
            assistant_msgs = [m for m in conversation.messages if m.role == "assistant"]
            system_msgs = [m for m in conversation.messages if m.role == "system"]

            lines.append("## Statistics")
            lines.append("")
            lines.append(f"- **User Messages:** {len(user_msgs)}")
            lines.append(f"- **Assistant Messages:** {len(assistant_msgs)}")
            lines.append(f"- **System Messages:** {len(system_msgs)}")
            lines.append(f"- **Avg Tokens/Message:** {conversation.get_token_count() / max(1, conversation.get_message_count()):.1f}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Add messages
    for i, msg in enumerate(conversation.messages, 1):
        role_emoji = {
            "system": "⚙️",
            "user": "👤",
            "assistant": "🤖"
        }.get(msg.role, "❓")

        lines.append(f"## {role_emoji} Message {i}: {msg.role.upper()}")
        lines.append("")
        lines.append(f"*Timestamp: {msg.timestamp}* | *Tokens: {msg.tokens}*")
        lines.append("")
        lines.append(msg.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    output = "\n".join(lines)

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(output)

    logger.info(
        f"Exported conversation to markdown: {filepath} "
        f"({conversation.get_message_count()} messages, {conversation.get_token_count()} tokens)"
    )


def export_to_json(
    conversation: "ConversationManager",
    filepath: str,
    pretty: bool = True,
    include_metadata: bool = True,
) -> None:
    """
    Export conversation as structured JSON.

    Creates a JSON file with complete conversation data including messages,
    metadata, and statistics. Supports both pretty-printed and compact formats.

    Args:
        conversation: ConversationManager to export
        filepath: Path to output file
        pretty: Whether to pretty-print JSON (default: True)
        include_metadata: Whether to include conversation metadata (default: True)

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Test message")
        >>> export_to_json(conv, "/tmp/chat.json")
        >>> with open("/tmp/chat.json") as f:
        ...     data = json.load(f)
        >>> "messages" in data
        True

    Notes:
        - Output includes all message data with timestamps and token counts
        - Metadata includes conversation creation time and configuration
        - Statistics provide token usage summary
    """
    # Build export data structure
    data: Dict[str, Any] = {
        "format_version": "1.0",
        "export_timestamp": datetime.utcnow().isoformat(),
        "messages": [msg.to_dict() for msg in conversation.messages],
    }

    if include_metadata:
        data["conversation_metadata"] = {
            "created_at": conversation.created_at,
            "max_context_tokens": conversation.max_context_tokens,
            "custom_metadata": conversation.metadata,
        }

        # Add statistics
        data["statistics"] = {
            "total_messages": conversation.get_message_count(),
            "total_tokens": conversation.get_token_count(),
            "message_breakdown": {
                "system": len([m for m in conversation.messages if m.role == "system"]),
                "user": len([m for m in conversation.messages if m.role == "user"]),
                "assistant": len([m for m in conversation.messages if m.role == "assistant"]),
            },
            "token_breakdown": {
                "system": sum(m.tokens for m in conversation.messages if m.role == "system"),
                "user": sum(m.tokens for m in conversation.messages if m.role == "user"),
                "assistant": sum(m.tokens for m in conversation.messages if m.role == "assistant"),
            },
        }

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)

    logger.info(
        f"Exported conversation to JSON: {filepath} "
        f"({conversation.get_message_count()} messages, {len(json.dumps(data))} bytes)"
    )


def export_to_text(
    conversation: "ConversationManager",
    filepath: str,
    include_metadata: bool = True,
    include_timestamps: bool = True,
    separator: str = "=" * 60,
) -> None:
    """
    Export conversation to plain text format.

    Creates a human-readable plain text file with optional metadata header
    and configurable separators between messages.

    Args:
        conversation: ConversationManager to export
        filepath: Path to output file
        include_metadata: Whether to include metadata header (default: True)
        include_timestamps: Whether to include message timestamps (default: True)
        separator: Separator line between messages (default: 60 equal signs)

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Hello world")
        >>> export_to_text(conv, "/tmp/chat.txt")
        >>> Path("/tmp/chat.txt").exists()
        True

    Notes:
        - Output is designed for easy human reading
        - Each message shows role, timestamp (optional), token count, and content
        - Metadata header provides conversation overview
    """
    lines = []

    if include_metadata:
        lines.append(separator)
        lines.append("CONVERSATION EXPORT")
        lines.append(separator)
        lines.append(f"Created: {conversation.created_at}")
        lines.append(f"Messages: {conversation.get_message_count()}")
        lines.append(f"Total Tokens: {conversation.get_token_count()}")
        lines.append(f"Max Context: {conversation.max_context_tokens} tokens")
        lines.append("")

        # Add custom metadata
        if conversation.metadata:
            lines.append("Custom Metadata:")
            for key, value in conversation.metadata.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append(separator)
        lines.append("")

    # Add messages
    for i, msg in enumerate(conversation.messages, 1):
        lines.append(f"[Message {i}] {msg.role.upper()}")

        if include_timestamps:
            lines.append(f"Timestamp: {msg.timestamp}")

        lines.append(f"Tokens: {msg.tokens}")
        lines.append("-" * len(separator))
        lines.append(msg.content)
        lines.append("")
        lines.append(separator)
        lines.append("")

    output = "\n".join(lines)

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(output)

    logger.info(
        f"Exported conversation to text: {filepath} "
        f"({conversation.get_message_count()} messages, {len(output)} characters)"
    )


def export_conversation(
    conversation: "ConversationManager",
    filepath: str,
    format: str = "auto",
    **kwargs
) -> None:
    """
    Export conversation to file with automatic format detection.

    Convenience function that automatically selects export format based on
    file extension, or uses explicitly specified format.

    Args:
        conversation: ConversationManager to export
        filepath: Path to output file
        format: Export format (auto, markdown, json, text). Default: auto
        **kwargs: Additional arguments passed to specific export function

    Raises:
        ValueError: If format is invalid or cannot be determined

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Test")
        >>> export_conversation(conv, "/tmp/chat.md")  # Auto-detects markdown
        >>> export_conversation(conv, "/tmp/chat.json")  # Auto-detects JSON
        >>> export_conversation(conv, "/tmp/chat.txt", format="text")  # Explicit

    Notes:
        - Format detection based on file extension: .md, .json, .txt
        - Falls back to text format if extension is unknown
        - All format-specific options can be passed via kwargs
    """
    # Determine format
    if format == "auto":
        ext = Path(filepath).suffix.lower()
        format_map = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".json": "json",
            ".txt": "text",
            ".log": "text",
        }
        format = format_map.get(ext, "text")

    # Export using appropriate function
    if format == "markdown":
        export_to_markdown(conversation, filepath, **kwargs)
    elif format == "json":
        export_to_json(conversation, filepath, **kwargs)
    elif format == "text":
        export_to_text(conversation, filepath, **kwargs)
    else:
        raise ValueError(
            f"Invalid export format: {format}. "
            f"Must be one of: markdown, json, text, auto"
        )

    logger.debug(f"Exported conversation using format: {format}")
