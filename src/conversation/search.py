"""
Conversation search and filtering utilities.

This module provides powerful search and filtering capabilities for conversations,
including full-text search, role filtering, and message retrieval.

Functions:
    search_messages: Full-text search across conversation
    get_message_by_id: Retrieve specific message by index
    filter_by_role: Filter messages by role
    find_messages_by_pattern: Regex pattern matching
    get_messages_in_range: Retrieve messages in time range
"""

import logging
import re
from datetime import datetime
from typing import List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .history import ConversationManager, Message

logger = logging.getLogger(__name__)


def search_messages(
    conversation: "ConversationManager",
    query: str,
    case_sensitive: bool = False,
    search_role: Optional[str] = None,
    max_results: Optional[int] = None,
) -> List["Message"]:
    """
    Search conversation history for messages matching query.

    Performs full-text search across message content, with optional
    role filtering and result limiting.

    Args:
        conversation: ConversationManager to search
        query: Search query string
        case_sensitive: Whether search should be case-sensitive (default: False)
        search_role: Optional role filter (system, user, assistant)
        max_results: Maximum number of results to return (default: unlimited)

    Returns:
        List of Message objects matching the query

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Hello world")
        >>> conv.add_message("assistant", "Hi there!")
        >>> conv.add_message("user", "How are you?")
        >>> results = search_messages(conv, "hello")
        >>> len(results)
        1
        >>> results[0].content
        'Hello world'

    Notes:
        - Search is substring-based by default
        - Returns messages in chronological order
        - Empty query returns all messages (filtered by role if specified)
    """
    if not query and search_role is None:
        logger.warning("search_messages called with no query and no role filter")
        return conversation.messages[:max_results] if max_results else conversation.messages

    results = []

    # Prepare query for case-insensitive search
    search_query = query if case_sensitive else query.lower()

    for msg in conversation.messages:
        # Apply role filter if specified
        if search_role and msg.role != search_role:
            continue

        # Search in content
        content = msg.content if case_sensitive else msg.content.lower()

        if search_query in content:
            results.append(msg)

            # Check max results limit
            if max_results and len(results) >= max_results:
                break

    logger.debug(
        f"Search found {len(results)} messages for query '{query}' "
        f"(case_sensitive={case_sensitive}, role={search_role})"
    )

    return results


def get_message_by_id(
    conversation: "ConversationManager",
    message_id: int
) -> Optional["Message"]:
    """
    Retrieve specific message by index.

    Args:
        conversation: ConversationManager to search
        message_id: Zero-based index of message to retrieve

    Returns:
        Message object if found, None otherwise

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "First message")
        >>> conv.add_message("assistant", "Second message")
        >>> msg = get_message_by_id(conv, 0)
        >>> msg.content
        'First message'
        >>> msg = get_message_by_id(conv, 1)
        >>> msg.role
        'assistant'
        >>> get_message_by_id(conv, 99) is None
        True

    Notes:
        - message_id is zero-based index
        - Returns None for out-of-bounds indices
        - Negative indices are supported (Python list indexing)
    """
    try:
        message = conversation.messages[message_id]
        logger.debug(f"Retrieved message #{message_id} (role={message.role})")
        return message
    except IndexError:
        logger.warning(
            f"Message ID {message_id} out of range "
            f"(conversation has {len(conversation.messages)} messages)"
        )
        return None


def filter_by_role(
    conversation: "ConversationManager",
    role: str
) -> List["Message"]:
    """
    Get all messages with a specific role.

    Args:
        conversation: ConversationManager to filter
        role: Role to filter by (system, user, assistant)

    Returns:
        List of messages with matching role

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("system", "System prompt")
        >>> conv.add_message("user", "User question")
        >>> conv.add_message("assistant", "Assistant response")
        >>> conv.add_message("user", "Another question")
        >>> user_msgs = filter_by_role(conv, "user")
        >>> len(user_msgs)
        2
        >>> all(m.role == "user" for m in user_msgs)
        True

    Notes:
        - Returns messages in chronological order
        - Returns empty list if no messages match
        - Role comparison is case-sensitive
    """
    filtered = [msg for msg in conversation.messages if msg.role == role]

    logger.debug(
        f"Filtered {len(filtered)} messages with role '{role}' "
        f"from {len(conversation.messages)} total"
    )

    return filtered


def find_messages_by_pattern(
    conversation: "ConversationManager",
    pattern: str,
    flags: int = 0,
    search_role: Optional[str] = None,
    max_results: Optional[int] = None,
) -> List["Message"]:
    """
    Find messages matching a regex pattern.

    Args:
        conversation: ConversationManager to search
        pattern: Regular expression pattern
        flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE)
        search_role: Optional role filter
        max_results: Maximum number of results to return

    Returns:
        List of messages matching the pattern

    Examples:
        >>> from src.conversation import ConversationManager
        >>> import re
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "What is Python?")
        >>> conv.add_message("assistant", "Python is a language")
        >>> conv.add_message("user", "Tell me about Java")
        >>> # Find questions
        >>> questions = find_messages_by_pattern(conv, r"\\?$")
        >>> len(questions)
        2
        >>> # Case-insensitive search
        >>> python_msgs = find_messages_by_pattern(
        ...     conv, r"python", flags=re.IGNORECASE
        ... )
        >>> len(python_msgs)
        2

    Notes:
        - Uses Python's re module for pattern matching
        - Returns messages in chronological order
        - Invalid patterns raise re.error exception
    """
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        raise ValueError(f"Invalid regex pattern: {e}") from e

    results = []

    for msg in conversation.messages:
        # Apply role filter if specified
        if search_role and msg.role != search_role:
            continue

        # Search for pattern in content
        if regex.search(msg.content):
            results.append(msg)

            # Check max results limit
            if max_results and len(results) >= max_results:
                break

    logger.debug(
        f"Pattern search found {len(results)} messages for pattern '{pattern}' "
        f"(role={search_role})"
    )

    return results


def get_messages_in_range(
    conversation: "ConversationManager",
    start: Optional[str] = None,
    end: Optional[str] = None,
    search_role: Optional[str] = None,
) -> List["Message"]:
    """
    Get messages within a timestamp range.

    Args:
        conversation: ConversationManager to search
        start: Start timestamp (ISO format) or None for beginning
        end: End timestamp (ISO format) or None for end
        search_role: Optional role filter

    Returns:
        List of messages within the timestamp range

    Examples:
        >>> from src.conversation import ConversationManager
        >>> from datetime import datetime, timedelta
        >>> conv = ConversationManager()
        >>> # Add messages over time
        >>> conv.add_message("user", "First")
        >>> import time; time.sleep(0.01)
        >>> middle_time = datetime.utcnow().isoformat()
        >>> time.sleep(0.01)
        >>> conv.add_message("user", "Second")
        >>> conv.add_message("user", "Third")
        >>> # Get messages after middle_time
        >>> recent = get_messages_in_range(conv, start=middle_time)
        >>> len(recent) >= 2
        True

    Notes:
        - Timestamps must be in ISO format
        - Start and end are inclusive
        - Returns empty list if timestamps are invalid
    """
    results = []

    # Parse timestamps
    start_dt = None
    end_dt = None

    try:
        if start:
            start_dt = datetime.fromisoformat(start)
        if end:
            end_dt = datetime.fromisoformat(end)
    except ValueError as e:
        logger.error(f"Invalid timestamp format: {e}")
        return []

    for msg in conversation.messages:
        # Apply role filter if specified
        if search_role and msg.role != search_role:
            continue

        try:
            msg_dt = datetime.fromisoformat(msg.timestamp)

            # Check if within range
            if start_dt and msg_dt < start_dt:
                continue
            if end_dt and msg_dt > end_dt:
                continue

            results.append(msg)

        except ValueError as e:
            logger.warning(f"Failed to parse message timestamp: {e}")
            continue

    logger.debug(
        f"Found {len(results)} messages in range "
        f"(start={start}, end={end}, role={search_role})"
    )

    return results


def get_messages_by_token_count(
    conversation: "ConversationManager",
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    search_role: Optional[str] = None,
) -> List["Message"]:
    """
    Get messages filtered by token count.

    Args:
        conversation: ConversationManager to search
        min_tokens: Minimum token count (inclusive)
        max_tokens: Maximum token count (inclusive)
        search_role: Optional role filter

    Returns:
        List of messages matching token count criteria

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Short")
        >>> conv.add_message("user", "This is a much longer message with more tokens")
        >>> # Get messages with more than 5 tokens
        >>> long_msgs = get_messages_by_token_count(conv, min_tokens=5)
        >>> len(long_msgs)
        1

    Notes:
        - Both min_tokens and max_tokens are inclusive
        - If neither limit is specified, returns all messages
    """
    results = []

    for msg in conversation.messages:
        # Apply role filter if specified
        if search_role and msg.role != search_role:
            continue

        # Check token count
        if min_tokens is not None and msg.tokens < min_tokens:
            continue
        if max_tokens is not None and msg.tokens > max_tokens:
            continue

        results.append(msg)

    logger.debug(
        f"Found {len(results)} messages with token count "
        f"in range ({min_tokens}, {max_tokens})"
    )

    return results


def filter_messages(
    conversation: "ConversationManager",
    predicate: Callable[["Message"], bool]
) -> List["Message"]:
    """
    Filter messages using a custom predicate function.

    Args:
        conversation: ConversationManager to filter
        predicate: Function that takes Message and returns bool

    Returns:
        List of messages where predicate returns True

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> conv.add_message("user", "Hello")
        >>> conv.add_message("assistant", "Hi!")
        >>> conv.add_message("user", "How are you?")
        >>> # Get messages with more than 5 characters
        >>> long_msgs = filter_messages(conv, lambda m: len(m.content) > 5)
        >>> len(long_msgs)
        2
        >>> # Get user messages with questions
        >>> questions = filter_messages(
        ...     conv, lambda m: m.role == "user" and "?" in m.content
        ... )
        >>> len(questions)
        1

    Notes:
        - Provides maximum flexibility for custom filtering
        - Predicate function should be efficient as it's called for each message
        - Returns messages in chronological order
    """
    results = [msg for msg in conversation.messages if predicate(msg)]

    logger.debug(f"Custom filter matched {len(results)} messages")

    return results


def get_conversation_context(
    conversation: "ConversationManager",
    message_id: int,
    context_before: int = 2,
    context_after: int = 2,
) -> List["Message"]:
    """
    Get message with surrounding context.

    Args:
        conversation: ConversationManager to search
        message_id: Index of target message
        context_before: Number of messages before target
        context_after: Number of messages after target

    Returns:
        List of messages including target and context

    Examples:
        >>> from src.conversation import ConversationManager
        >>> conv = ConversationManager()
        >>> for i in range(10):
        ...     conv.add_message("user", f"Message {i}")
        >>> # Get message 5 with 2 messages before and after
        >>> context = get_conversation_context(conv, 5, 2, 2)
        >>> len(context)
        5
        >>> context[2].content
        'Message 5'

    Notes:
        - Returns fewer messages if near conversation boundaries
        - Target message is included in results
        - Returns empty list if message_id is invalid
    """
    if message_id < 0 or message_id >= len(conversation.messages):
        logger.warning(f"Invalid message_id: {message_id}")
        return []

    start_idx = max(0, message_id - context_before)
    end_idx = min(len(conversation.messages), message_id + context_after + 1)

    context = conversation.messages[start_idx:end_idx]

    logger.debug(
        f"Retrieved context for message {message_id}: "
        f"{len(context)} messages (indices {start_idx}-{end_idx-1})"
    )

    return context
