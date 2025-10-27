#!/usr/bin/env python3
"""
Example: Conversation Search

This script demonstrates the search and filtering functionality,
showing how to find specific messages in conversations.
"""

import sys
import re
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import (
    ConversationManager,
    search_messages,
    get_message_by_id,
    filter_by_role,
    find_messages_by_pattern,
    get_messages_by_token_count,
    filter_messages,
    get_conversation_context,
)


def create_tech_conversation():
    """Create a conversation about various tech topics."""
    conv = ConversationManager()

    conv.add_message("system", "You are a tech expert.")

    # Python discussion
    conv.add_message("user", "What is Python used for?")
    conv.add_message("assistant", "Python is used for web development, data science, AI, and automation.")

    # JavaScript discussion
    conv.add_message("user", "Tell me about JavaScript.")
    conv.add_message("assistant", "JavaScript is primarily used for web development, both frontend and backend.")

    # Data science discussion
    conv.add_message("user", "What are the best tools for data science?")
    conv.add_message("assistant", "Python with libraries like pandas, numpy, and scikit-learn are excellent for data science.")

    # Cloud computing
    conv.add_message("user", "How does cloud computing work?")
    conv.add_message("assistant", "Cloud computing provides on-demand computing resources over the internet.")

    # Python again
    conv.add_message("user", "Is Python good for beginners?")
    conv.add_message("assistant", "Yes! Python's simple syntax makes it ideal for beginners learning programming.")

    return conv


def main():
    """Demonstrate search functionality."""
    print("=== Conversation Search Demo ===\n")

    # Create sample conversation
    print("Creating sample conversation...")
    conv = create_tech_conversation()
    print(f"Created conversation with {conv.get_message_count()} messages\n")

    # Basic text search
    print("1. Basic Text Search")
    print("-" * 60)
    python_msgs = search_messages(conv, "Python")
    print(f"Found {len(python_msgs)} messages mentioning 'Python':")
    for i, msg in enumerate(python_msgs, 1):
        preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"  {i}. [{msg.role}] {preview}")
    print("\n")

    # Case-insensitive search
    print("2. Case-Insensitive Search")
    print("-" * 60)
    cloud_msgs = search_messages(conv, "CLOUD", case_sensitive=False)
    print(f"Found {len(cloud_msgs)} messages about 'CLOUD' (case-insensitive):")
    for msg in cloud_msgs:
        print(f"  [{msg.role}] {msg.content}")
    print("\n")

    # Search by role
    print("3. Search Within Specific Role")
    print("-" * 60)
    user_python = search_messages(conv, "Python", search_role="user")
    print(f"Found {len(user_python)} user messages about Python:")
    for msg in user_python:
        print(f"  {msg.content}")
    print("\n")

    # Get message by ID
    print("4. Get Message by ID")
    print("-" * 60)
    msg = get_message_by_id(conv, 3)
    if msg:
        print(f"Message #3:")
        print(f"  Role: {msg.role}")
        print(f"  Content: {msg.content}")
        print(f"  Tokens: {msg.tokens}")
    print("\n")

    # Filter by role
    print("5. Filter by Role")
    print("-" * 60)
    user_msgs = filter_by_role(conv, "user")
    assistant_msgs = filter_by_role(conv, "assistant")
    print(f"User messages: {len(user_msgs)}")
    print(f"Assistant messages: {len(assistant_msgs)}")
    print("\nAll user questions:")
    for i, msg in enumerate(user_msgs, 1):
        print(f"  {i}. {msg.content}")
    print("\n")

    # Pattern matching - find questions
    print("6. Pattern Matching (Questions)")
    print("-" * 60)
    questions = find_messages_by_pattern(conv, r"\?$")
    print(f"Found {len(questions)} questions:")
    for msg in questions:
        print(f"  {msg.content}")
    print("\n")

    # Pattern matching - case insensitive
    print("7. Case-Insensitive Pattern Matching")
    print("-" * 60)
    web_msgs = find_messages_by_pattern(conv, r"web", flags=re.IGNORECASE)
    print(f"Found {len(web_msgs)} messages mentioning 'web':")
    for msg in web_msgs:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  [{msg.role}] {preview}")
    print("\n")

    # Filter by token count
    print("8. Filter by Token Count")
    print("-" * 60)
    long_msgs = get_messages_by_token_count(conv, min_tokens=10)
    short_msgs = get_messages_by_token_count(conv, max_tokens=10)
    print(f"Messages with 10+ tokens: {len(long_msgs)}")
    print(f"Messages with ≤10 tokens: {len(short_msgs)}")
    print("\nLongest messages:")
    for msg in sorted(long_msgs, key=lambda m: m.tokens, reverse=True)[:3]:
        print(f"  {msg.tokens} tokens: {msg.content[:50]}...")
    print("\n")

    # Custom filter
    print("9. Custom Filter (User questions about Python)")
    print("-" * 60)
    python_questions = filter_messages(
        conv,
        lambda m: m.role == "user" and "Python" in m.content and "?" in m.content
    )
    print(f"Found {len(python_questions)} user questions about Python:")
    for msg in python_questions:
        print(f"  {msg.content}")
    print("\n")

    # Get message with context
    print("10. Get Message with Context")
    print("-" * 60)
    target_id = 5
    context = get_conversation_context(conv, target_id, context_before=1, context_after=1)
    print(f"Message #{target_id} with context (1 before, 1 after):")
    for i, msg in enumerate(context):
        marker = ">>>" if i == 1 else "   "  # Mark target message
        print(f"{marker} [{msg.role}] {msg.content}")
    print("\n")

    # Limit search results
    print("11. Limit Search Results")
    print("-" * 60)
    limited = search_messages(conv, "Python", max_results=2)
    print(f"First 2 messages about Python:")
    for msg in limited:
        print(f"  [{msg.role}] {msg.content}")
    print("\n")

    print("✓ Search demo complete!")


if __name__ == "__main__":
    main()
