"""
Conversation management module.

This module provides tools for managing multi-turn conversations, including:
- Message history management with Harmony channel support
- Context window tracking and truncation
- Token counting and estimation
- Conversation persistence (save/load) with channel preservation
- Export to multiple formats (markdown, JSON, text)
- Statistics tracking and analytics
- Advanced search and filtering

Main exports:
- ConversationManager: Core conversation management
- Message: Single message dataclass with Harmony channels
- ContextWindowTracker: Context window management
- ConversationPersistence: Save/load conversations
- ConversationStats: Statistics tracking
- Token utilities: Token counting functions
- Export functions: Export to various formats
- Search functions: Advanced message search and filtering

Harmony Integration (Wave 3):
- Messages can store multi-channel Harmony responses (final, analysis, commentary)
- ConversationManager.add_harmony_response() for adding parsed Harmony responses
- ConversationManager.get_messages_for_harmony() for HarmonyPromptBuilder integration
- Backward compatible with non-Harmony messages
"""

from .history import ConversationManager, Message
from .context import ContextWindowTracker, TruncationStrategy
from .persistence import ConversationPersistence
from .tokens import (
    estimate_tokens,
    count_message_tokens,
    count_conversation_tokens,
    estimate_tokens_remaining,
    truncate_to_token_limit,
    TokenCounter,
)

# Wave 2: Export functionality
from .export import (
    export_to_markdown,
    export_to_json,
    export_to_text,
    export_conversation,
)

# Wave 2: Statistics tracking
from .stats import (
    ConversationStats,
    MessageStats,
    compute_statistics,
    compare_conversations,
)

# Wave 2: Search and filtering
from .search import (
    search_messages,
    get_message_by_id,
    filter_by_role,
    find_messages_by_pattern,
    get_messages_in_range,
    get_messages_by_token_count,
    filter_messages,
    get_conversation_context,
)

__all__ = [
    # Core classes
    "ConversationManager",
    "Message",
    "ContextWindowTracker",
    "ConversationPersistence",
    "TokenCounter",
    # Enums
    "TruncationStrategy",
    # Token utilities
    "estimate_tokens",
    "count_message_tokens",
    "count_conversation_tokens",
    "estimate_tokens_remaining",
    "truncate_to_token_limit",
    # Export functions (Wave 2)
    "export_to_markdown",
    "export_to_json",
    "export_to_text",
    "export_conversation",
    # Statistics (Wave 2)
    "ConversationStats",
    "MessageStats",
    "compute_statistics",
    "compare_conversations",
    # Search and filtering (Wave 2)
    "search_messages",
    "get_message_by_id",
    "filter_by_role",
    "find_messages_by_pattern",
    "get_messages_in_range",
    "get_messages_by_token_count",
    "filter_messages",
    "get_conversation_context",
]

__version__ = "3.0.0"  # Harmony integration with channel support
