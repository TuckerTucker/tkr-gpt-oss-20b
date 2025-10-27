"""
Unit tests for Wave 2 conversation features.

Tests for export, statistics, and search functionality.
"""

import json
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
import time
import re

from src.conversation import (
    ConversationManager,
    Message,
    # Export functions
    export_to_markdown,
    export_to_json,
    export_to_text,
    export_conversation,
    # Statistics
    ConversationStats,
    compute_statistics,
    compare_conversations,
    # Search functions
    search_messages,
    get_message_by_id,
    filter_by_role,
    find_messages_by_pattern,
    get_messages_in_range,
    get_messages_by_token_count,
    filter_messages,
    get_conversation_context,
)


# =============================================================================
# Export Tests
# =============================================================================


class TestExport:
    """Test conversation export functionality."""

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation for testing."""
        conv = ConversationManager(
            max_context_tokens=2048,
            metadata={"model": "test-model", "version": "1.0"}
        )
        conv.add_message("system", "You are a helpful assistant.")
        conv.add_message("user", "What is Python?")
        conv.add_message("assistant", "Python is a high-level programming language.")
        conv.add_message("user", "Tell me more about its features.")
        return conv

    def test_export_to_markdown(self, sample_conversation, tmp_path):
        """Test exporting conversation to markdown format."""
        output_file = tmp_path / "conversation.md"
        export_to_markdown(sample_conversation, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()

        # Check for markdown headers
        assert "# Conversation Export" in content
        assert "## Statistics" in content

        # Check for message content
        assert "What is Python?" in content
        assert "Python is a high-level programming language" in content

        # Check for emojis
        assert "⚙️" in content  # System emoji
        assert "👤" in content  # User emoji
        assert "🤖" in content  # Assistant emoji

    def test_export_to_markdown_no_metadata(self, sample_conversation, tmp_path):
        """Test exporting without metadata."""
        output_file = tmp_path / "conversation.md"
        export_to_markdown(
            sample_conversation,
            str(output_file),
            include_metadata=False
        )

        content = output_file.read_text()
        assert "# Conversation Export" not in content
        assert "What is Python?" in content

    def test_export_to_json(self, sample_conversation, tmp_path):
        """Test exporting conversation to JSON format."""
        output_file = tmp_path / "conversation.json"
        export_to_json(sample_conversation, str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        # Check structure
        assert "format_version" in data
        assert "export_timestamp" in data
        assert "messages" in data
        assert "conversation_metadata" in data
        assert "statistics" in data

        # Check messages
        assert len(data["messages"]) == 4
        assert data["messages"][1]["content"] == "What is Python?"

        # Check statistics
        stats = data["statistics"]
        assert stats["total_messages"] == 4
        assert stats["message_breakdown"]["user"] == 2
        assert stats["message_breakdown"]["assistant"] == 1
        assert stats["message_breakdown"]["system"] == 1

    def test_export_to_json_compact(self, sample_conversation, tmp_path):
        """Test compact JSON export."""
        output_file = tmp_path / "conversation.json"
        export_to_json(sample_conversation, str(output_file), pretty=False)

        content = output_file.read_text()
        # Compact JSON should have fewer newlines
        assert content.count("\n") < 10

    def test_export_to_text(self, sample_conversation, tmp_path):
        """Test exporting conversation to plain text format."""
        output_file = tmp_path / "conversation.txt"
        export_to_text(sample_conversation, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()

        # Check for header
        assert "CONVERSATION EXPORT" in content

        # Check for messages
        assert "[Message 1] SYSTEM" in content
        assert "[Message 2] USER" in content
        assert "What is Python?" in content

        # Check for separators
        assert "=" * 60 in content

    def test_export_to_text_no_timestamps(self, sample_conversation, tmp_path):
        """Test text export without timestamps."""
        output_file = tmp_path / "conversation.txt"
        export_to_text(
            sample_conversation,
            str(output_file),
            include_timestamps=False
        )

        content = output_file.read_text()
        assert "Timestamp:" not in content

    def test_export_conversation_auto_detect(self, sample_conversation, tmp_path):
        """Test automatic format detection based on file extension."""
        # Test markdown
        md_file = tmp_path / "chat.md"
        export_conversation(sample_conversation, str(md_file))
        assert md_file.exists()
        assert "# Conversation Export" in md_file.read_text()

        # Test JSON
        json_file = tmp_path / "chat.json"
        export_conversation(sample_conversation, str(json_file))
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert "messages" in data

        # Test text
        txt_file = tmp_path / "chat.txt"
        export_conversation(sample_conversation, str(txt_file))
        assert txt_file.exists()
        assert "CONVERSATION EXPORT" in txt_file.read_text()

    def test_export_conversation_explicit_format(self, sample_conversation, tmp_path):
        """Test explicit format specification."""
        output_file = tmp_path / "output.dat"
        export_conversation(sample_conversation, str(output_file), format="json")

        data = json.loads(output_file.read_text())
        assert "messages" in data

    def test_export_invalid_format(self, sample_conversation, tmp_path):
        """Test export with invalid format raises error."""
        output_file = tmp_path / "output.txt"
        with pytest.raises(ValueError, match="Invalid export format"):
            export_conversation(sample_conversation, str(output_file), format="xml")


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test conversation statistics tracking."""

    @pytest.fixture
    def simple_conversation(self):
        """Create a simple conversation for stats testing."""
        conv = ConversationManager()
        conv.add_message("system", "You are helpful.")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        return conv

    def test_compute_statistics(self, simple_conversation):
        """Test computing basic statistics."""
        stats = compute_statistics(simple_conversation)

        assert stats.total_messages == 3
        assert stats.total_tokens > 0
        assert stats.message_breakdown["system"] == 1
        assert stats.message_breakdown["user"] == 1
        assert stats.message_breakdown["assistant"] == 1

    def test_conversation_stats_from_conversation(self):
        """Test creating ConversationStats from conversation."""
        conv = ConversationManager()
        for i in range(5):
            conv.add_message("user", f"Message {i}")
            conv.add_message("assistant", f"Response {i}")

        stats = ConversationStats.from_conversation(conv)

        assert stats.total_messages == 10
        assert stats.message_breakdown["user"] == 5
        assert stats.message_breakdown["assistant"] == 5
        assert stats.avg_tokens_per_message > 0

    def test_stats_role_percentage(self):
        """Test calculating role percentages."""
        conv = ConversationManager()
        conv.add_message("user", "Test 1")
        conv.add_message("user", "Test 2")
        conv.add_message("user", "Test 3")
        conv.add_message("assistant", "Response")

        stats = ConversationStats.from_conversation(conv)

        assert stats.get_role_percentage("user") == 75.0
        assert stats.get_role_percentage("assistant") == 25.0
        assert stats.get_role_percentage("system") == 0.0

    def test_stats_token_percentage(self):
        """Test calculating token percentages."""
        conv = ConversationManager()
        conv.add_message("user", "Short")
        conv.add_message("assistant", "This is a much longer message with many more tokens")

        stats = ConversationStats.from_conversation(conv)

        user_pct = stats.get_token_percentage("user")
        assistant_pct = stats.get_token_percentage("assistant")

        assert assistant_pct > user_pct
        assert abs((user_pct + assistant_pct) - 100.0) < 0.1

    def test_stats_token_efficiency(self):
        """Test token efficiency calculation."""
        conv = ConversationManager()
        conv.add_message("user", "Hello world")

        stats = ConversationStats.from_conversation(conv)
        efficiency = stats.get_token_efficiency()

        assert 0 < efficiency < 1  # Should be between 0 and 1

    def test_stats_average_response_time(self):
        """Test average response time calculation."""
        conv = ConversationManager()
        conv.add_message("user", "Question 1")
        time.sleep(0.01)
        conv.add_message("assistant", "Answer 1")
        time.sleep(0.01)
        conv.add_message("user", "Question 2")
        time.sleep(0.01)
        conv.add_message("assistant", "Answer 2")

        stats = ConversationStats.from_conversation(conv)
        avg_response = stats.get_average_response_time()

        assert avg_response is not None
        assert avg_response >= 0

    def test_stats_token_usage_trend(self):
        """Test token usage trend over time."""
        conv = ConversationManager()
        for i in range(3):
            conv.add_message("user", f"Message {i}")

        stats = ConversationStats.from_conversation(conv)
        trend = stats.get_token_usage_trend()

        assert len(trend) == 3
        # Cumulative tokens should increase
        assert trend[1]["cumulative_tokens"] >= trend[0]["cumulative_tokens"]
        assert trend[2]["cumulative_tokens"] >= trend[1]["cumulative_tokens"]

    def test_stats_to_dict(self, simple_conversation):
        """Test converting stats to dictionary."""
        stats = ConversationStats.from_conversation(simple_conversation)
        data = stats.to_dict()

        assert "total_messages" in data
        assert "total_tokens" in data
        assert "message_breakdown" in data
        assert data["total_messages"] == 3

    def test_stats_to_summary(self, simple_conversation):
        """Test generating human-readable summary."""
        stats = ConversationStats.from_conversation(simple_conversation)
        summary = stats.to_summary()

        assert "=== Conversation Statistics ===" in summary
        assert "Total Messages: 3" in summary
        assert "Message Breakdown:" in summary
        assert "Token Breakdown:" in summary

    def test_compare_conversations(self):
        """Test comparing two conversations."""
        conv1 = ConversationManager()
        conv1.add_message("user", "Short")

        conv2 = ConversationManager()
        for i in range(5):
            conv2.add_message("user", "Longer message here")

        comparison = compare_conversations(conv1, conv2)

        assert "conversation_1" in comparison
        assert "conversation_2" in comparison
        assert "differences" in comparison
        assert "ratios" in comparison

        assert comparison["differences"]["message_difference"] == 4
        assert comparison["ratios"]["message_ratio"] == 5.0

    def test_stats_empty_conversation(self):
        """Test statistics on empty conversation."""
        conv = ConversationManager()
        stats = ConversationStats.from_conversation(conv)

        assert stats.total_messages == 0
        assert stats.total_tokens == 0
        assert stats.avg_tokens_per_message == 0.0


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test conversation search functionality."""

    @pytest.fixture
    def search_conversation(self):
        """Create a conversation for search testing."""
        conv = ConversationManager()
        conv.add_message("system", "You are a helpful assistant.")
        conv.add_message("user", "What is Python?")
        conv.add_message("assistant", "Python is a programming language.")
        conv.add_message("user", "Tell me about Java.")
        conv.add_message("assistant", "Java is also a programming language.")
        conv.add_message("user", "Which one is better?")
        return conv

    def test_search_messages_basic(self, search_conversation):
        """Test basic message search."""
        results = search_messages(search_conversation, "Python")

        assert len(results) == 2
        assert any("What is Python?" in msg.content for msg in results)
        assert any("Python is a programming language" in msg.content for msg in results)

    def test_search_messages_case_insensitive(self, search_conversation):
        """Test case-insensitive search."""
        results = search_messages(search_conversation, "python")

        assert len(results) == 2

    def test_search_messages_case_sensitive(self, search_conversation):
        """Test case-sensitive search."""
        results = search_messages(search_conversation, "Python", case_sensitive=True)

        assert len(results) == 2

        results = search_messages(search_conversation, "python", case_sensitive=True)
        assert len(results) == 0

    def test_search_messages_by_role(self, search_conversation):
        """Test searching within specific role."""
        results = search_messages(
            search_conversation,
            "language",
            search_role="assistant"
        )

        assert len(results) == 2
        assert all(msg.role == "assistant" for msg in results)

    def test_search_messages_max_results(self, search_conversation):
        """Test limiting search results."""
        results = search_messages(
            search_conversation,
            "language",
            max_results=1
        )

        assert len(results) == 1

    def test_get_message_by_id(self, search_conversation):
        """Test retrieving message by ID."""
        msg = get_message_by_id(search_conversation, 1)

        assert msg is not None
        assert msg.content == "What is Python?"

    def test_get_message_by_id_negative_index(self, search_conversation):
        """Test negative indexing."""
        msg = get_message_by_id(search_conversation, -1)

        assert msg is not None
        assert "better" in msg.content

    def test_get_message_by_id_invalid(self, search_conversation):
        """Test invalid message ID."""
        msg = get_message_by_id(search_conversation, 999)

        assert msg is None

    def test_filter_by_role(self, search_conversation):
        """Test filtering by role."""
        user_msgs = filter_by_role(search_conversation, "user")
        assistant_msgs = filter_by_role(search_conversation, "assistant")
        system_msgs = filter_by_role(search_conversation, "system")

        assert len(user_msgs) == 3
        assert len(assistant_msgs) == 2
        assert len(system_msgs) == 1

        assert all(msg.role == "user" for msg in user_msgs)

    def test_find_messages_by_pattern(self, search_conversation):
        """Test regex pattern search."""
        # Find questions (messages ending with ?)
        questions = find_messages_by_pattern(search_conversation, r"\?$")

        # The search_conversation fixture has user questions with "?"
        # "What is Python?" and "Which one is better?" end with ?
        assert len(questions) >= 2
        assert all("?" in msg.content for msg in questions)

    def test_find_messages_by_pattern_case_insensitive(self, search_conversation):
        """Test case-insensitive regex search."""
        results = find_messages_by_pattern(
            search_conversation,
            r"PYTHON",
            flags=re.IGNORECASE
        )

        assert len(results) == 2

    def test_find_messages_by_pattern_invalid(self, search_conversation):
        """Test invalid regex pattern."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            find_messages_by_pattern(search_conversation, r"[invalid(")

    def test_get_messages_in_range(self):
        """Test retrieving messages in time range."""
        conv = ConversationManager()

        # Add messages with delays
        conv.add_message("user", "First")
        time.sleep(0.01)
        middle_time = datetime.utcnow().isoformat()
        time.sleep(0.01)
        conv.add_message("user", "Second")
        conv.add_message("user", "Third")

        # Get messages after middle time
        recent = get_messages_in_range(conv, start=middle_time)

        assert len(recent) >= 2

    def test_get_messages_by_token_count(self, search_conversation):
        """Test filtering by token count."""
        # Find messages with more tokens (longer messages)
        long_msgs = get_messages_by_token_count(search_conversation, min_tokens=5)

        assert len(long_msgs) > 0
        assert all(msg.tokens >= 5 for msg in long_msgs)

    def test_get_messages_by_token_range(self, search_conversation):
        """Test filtering by token range."""
        msgs = get_messages_by_token_count(
            search_conversation,
            min_tokens=3,
            max_tokens=10
        )

        assert all(3 <= msg.tokens <= 10 for msg in msgs)

    def test_filter_messages_custom(self, search_conversation):
        """Test custom message filtering."""
        # Find user messages with more than 10 characters
        results = filter_messages(
            search_conversation,
            lambda m: m.role == "user" and len(m.content) > 10
        )

        assert len(results) > 0
        assert all(msg.role == "user" for msg in results)
        assert all(len(msg.content) > 10 for msg in results)

    def test_get_conversation_context(self):
        """Test retrieving message with context."""
        conv = ConversationManager()
        for i in range(10):
            conv.add_message("user", f"Message {i}")

        # Get message 5 with 2 messages before and after
        context = get_conversation_context(conv, 5, 2, 2)

        assert len(context) == 5
        assert context[0].content == "Message 3"
        assert context[2].content == "Message 5"  # Target message
        assert context[4].content == "Message 7"

    def test_get_conversation_context_boundary(self):
        """Test context at conversation boundaries."""
        conv = ConversationManager()
        for i in range(5):
            conv.add_message("user", f"Message {i}")

        # Get first message with context
        context = get_conversation_context(conv, 0, 2, 2)
        assert len(context) == 3  # Can't go before first message
        assert context[0].content == "Message 0"

        # Get last message with context
        context = get_conversation_context(conv, 4, 2, 2)
        assert len(context) == 3  # Can't go after last message
        assert context[-1].content == "Message 4"

    def test_get_conversation_context_invalid(self):
        """Test context with invalid message ID."""
        conv = ConversationManager()
        conv.add_message("user", "Test")

        context = get_conversation_context(conv, 999, 2, 2)
        assert len(context) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Test integration between features."""

    def test_export_with_statistics(self, tmp_path):
        """Test exporting conversation with statistics."""
        conv = ConversationManager()
        for i in range(10):
            conv.add_message("user", f"Question {i}")
            conv.add_message("assistant", f"Answer {i}")

        # Export to JSON with stats
        output_file = tmp_path / "with_stats.json"
        export_to_json(conv, str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        # Verify statistics are included
        stats = data["statistics"]
        assert stats["total_messages"] == 20
        assert stats["message_breakdown"]["user"] == 10
        assert stats["message_breakdown"]["assistant"] == 10

    def test_search_and_export(self, tmp_path):
        """Test searching and exporting results."""
        conv = ConversationManager()
        conv.add_message("user", "Tell me about Python")
        conv.add_message("assistant", "Python is great")
        conv.add_message("user", "Tell me about Java")
        conv.add_message("assistant", "Java is also good")

        # Search for Python-related messages
        results = search_messages(conv, "Python")

        # Create new conversation with results
        filtered_conv = ConversationManager()
        for msg in results:
            filtered_conv.messages.append(msg)

        # Export filtered conversation
        output_file = tmp_path / "filtered.md"
        export_to_markdown(filtered_conv, str(output_file))

        content = output_file.read_text()
        assert "Python" in content
        assert "Java" not in content

    def test_stats_on_filtered_conversation(self):
        """Test computing statistics on filtered conversation."""
        conv = ConversationManager()
        for i in range(20):
            conv.add_message("user", f"Message {i}")
            conv.add_message("assistant", f"Response {i}")

        # Filter to only user messages
        user_msgs = filter_by_role(conv, "user")

        # Create new conversation with filtered messages
        filtered_conv = ConversationManager()
        filtered_conv.messages = user_msgs

        # Compute stats
        stats = ConversationStats.from_conversation(filtered_conv)

        assert stats.total_messages == 20
        assert stats.message_breakdown["user"] == 20
        assert "assistant" not in stats.message_breakdown
