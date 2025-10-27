#!/usr/bin/env python3
"""
Example: Conversation Statistics

This script demonstrates the statistics tracking functionality,
showing how to compute and analyze conversation metrics.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import (
    ConversationManager,
    ConversationStats,
    compute_statistics,
    compare_conversations,
)


def create_short_conversation():
    """Create a short conversation."""
    conv = ConversationManager()
    conv.add_message("system", "You are helpful.")
    conv.add_message("user", "Hi")
    conv.add_message("assistant", "Hello!")
    return conv


def create_long_conversation():
    """Create a longer conversation."""
    conv = ConversationManager()
    conv.add_message("system", "You are a helpful assistant.")

    questions = [
        "What is machine learning?",
        "How does it differ from traditional programming?",
        "What are some common algorithms?",
        "Can you explain neural networks?",
        "What is deep learning?",
    ]

    for question in questions:
        conv.add_message("user", question)
        conv.add_message(
            "assistant",
            f"Here's a detailed explanation about your question: {question}. "
            "This is a comprehensive response that covers multiple aspects "
            "of the topic with examples and clarifications."
        )

    return conv


def main():
    """Demonstrate statistics functionality."""
    print("=== Conversation Statistics Demo ===\n")

    # Create conversations
    print("Creating sample conversations...\n")
    short_conv = create_short_conversation()
    long_conv = create_long_conversation()

    # Compute statistics for short conversation
    print("1. Short Conversation Statistics")
    print("-" * 60)
    short_stats = compute_statistics(short_conv)
    print(short_stats.to_summary())
    print("\n")

    # Compute statistics for long conversation
    print("2. Long Conversation Statistics")
    print("-" * 60)
    long_stats = ConversationStats.from_conversation(long_conv)
    print(long_stats.to_summary())
    print("\n")

    # Analyze role percentages
    print("3. Role Distribution Analysis")
    print("-" * 60)
    print(f"Short conversation:")
    print(f"  System: {short_stats.get_role_percentage('system'):.1f}%")
    print(f"  User: {short_stats.get_role_percentage('user'):.1f}%")
    print(f"  Assistant: {short_stats.get_role_percentage('assistant'):.1f}%")
    print()
    print(f"Long conversation:")
    print(f"  System: {long_stats.get_role_percentage('system'):.1f}%")
    print(f"  User: {long_stats.get_role_percentage('user'):.1f}%")
    print(f"  Assistant: {long_stats.get_role_percentage('assistant'):.1f}%")
    print("\n")

    # Token efficiency
    print("4. Token Efficiency")
    print("-" * 60)
    short_efficiency = short_stats.get_token_efficiency()
    long_efficiency = long_stats.get_token_efficiency()
    print(f"Short conversation: {short_efficiency:.3f} tokens/character")
    print(f"Long conversation: {long_efficiency:.3f} tokens/character")
    print("\n")

    # Token usage trend
    print("5. Token Usage Trend (Long Conversation)")
    print("-" * 60)
    trend = long_stats.get_token_usage_trend()
    print(f"{'Message':<10} {'Role':<12} {'Tokens':<10} {'Cumulative':<12}")
    print("-" * 60)
    for i, entry in enumerate(trend[:5], 1):  # Show first 5
        print(f"{i:<10} {entry['role']:<12} {entry['tokens']:<10} {entry['cumulative_tokens']:<12}")
    print(f"... ({len(trend) - 5} more messages)")
    print("\n")

    # Compare conversations
    print("6. Conversation Comparison")
    print("-" * 60)
    comparison = compare_conversations(short_conv, long_conv)
    print("Short vs Long conversation:")
    print(f"  Message difference: {comparison['differences']['message_difference']}")
    print(f"  Token difference: {comparison['differences']['token_difference']}")
    print(f"  Message ratio: {comparison['ratios']['message_ratio']:.2f}x")
    print(f"  Token ratio: {comparison['ratios']['token_ratio']:.2f}x")
    print("\n")

    # Statistics as dictionary
    print("7. Statistics as Dictionary")
    print("-" * 60)
    stats_dict = short_stats.to_dict()
    print("Short conversation stats dict keys:")
    for key in stats_dict.keys():
        print(f"  - {key}")
    print("\n")

    # Average response time (if available)
    print("8. Response Time Analysis")
    print("-" * 60)
    avg_response = long_stats.get_average_response_time()
    if avg_response is not None:
        print(f"Average response time: {avg_response:.4f} seconds")
    else:
        print("Response time data not available (messages too close in time)")
    print("\n")

    print("✓ Statistics demo complete!")


if __name__ == "__main__":
    main()
