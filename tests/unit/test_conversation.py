"""
Unit tests for conversation management and prompt formatting.

Tests ConversationManager, Message handling, and prompt templates.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Test suite for Message class."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement Message")
    def test_message_creation(self):
        """Test creating a Message with required fields."""
        from src.conversation import Message

        msg = Message(
            role="user",
            content="Hello, world!",
            timestamp="2025-01-01T12:00:00",
            tokens=3
        )

        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.timestamp == "2025-01-01T12:00:00"
        assert msg.tokens == 3

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement Message")
    def test_message_valid_roles(self):
        """Test Message accepts valid role types."""
        from src.conversation import Message

        valid_roles = ["system", "user", "assistant"]

        for role in valid_roles:
            msg = Message(
                role=role,
                content="Test",
                timestamp="2025-01-01T12:00:00",
                tokens=1
            )
            assert msg.role == role

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement Message")
    def test_message_serialization(self):
        """Test Message can be serialized to dict/JSON."""
        from src.conversation import Message

        msg = Message(
            role="user",
            content="Test message",
            timestamp="2025-01-01T12:00:00",
            tokens=2
        )

        # Should be serializable
        msg_dict = {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "tokens": msg.tokens
        }

        json_str = json.dumps(msg_dict)
        assert "user" in json_str
        assert "Test message" in json_str


# =============================================================================
# ConversationManager Tests
# =============================================================================

class TestConversationManager:
    """Test suite for ConversationManager class."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_conversation_manager_initialization(self):
        """Test ConversationManager initializes with context limit."""
        from src.conversation import ConversationManager

        manager = ConversationManager(max_context_tokens=2048)

        assert manager.max_context_tokens == 2048
        assert len(manager.get_history()) == 0

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_add_message_creates_message(self):
        """Test add_message() creates and returns Message."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        msg = manager.add_message("user", "Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tokens > 0
        assert msg.timestamp is not None

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_add_message_to_history(self):
        """Test messages are added to history."""
        from src.conversation import ConversationManager

        manager = ConversationManager()

        manager.add_message("system", "You are helpful")
        manager.add_message("user", "What is AI?")
        manager.add_message("assistant", "AI is...")

        history = manager.get_history()
        assert len(history) == 3
        assert history[0].role == "system"
        assert history[1].role == "user"
        assert history[2].role == "assistant"

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_get_history_returns_copy(self):
        """Test get_history() returns copy, not reference."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Test")

        history1 = manager.get_history()
        history2 = manager.get_history()

        # Should be different list objects
        assert history1 is not history2
        # But contain same messages
        assert len(history1) == len(history2)

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_clear_removes_all_messages(self):
        """Test clear() removes all messages."""
        from src.conversation import ConversationManager

        manager = ConversationManager()

        manager.add_message("user", "Test 1")
        manager.add_message("user", "Test 2")
        assert len(manager.get_history()) == 2

        manager.clear()
        assert len(manager.get_history()) == 0

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement ConversationManager")
    def test_get_token_count(self):
        """Test get_token_count() returns total tokens."""
        from src.conversation import ConversationManager

        manager = ConversationManager()

        # Each message has some tokens
        manager.add_message("user", "Hello")  # ~1 token
        manager.add_message("assistant", "Hi there")  # ~2 tokens

        total_tokens = manager.get_token_count()
        assert total_tokens > 0
        assert isinstance(total_tokens, int)


# =============================================================================
# Prompt Formatting Tests
# =============================================================================

class TestPromptFormatting:
    """Test suite for prompt formatting."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement format_prompt")
    def test_format_prompt_chatml(self):
        """Test format_prompt() with chatml template."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("system", "You are helpful")
        manager.add_message("user", "Hello")

        prompt = manager.format_prompt(template="chatml")

        assert "<|system|>" in prompt or "system" in prompt.lower()
        assert "<|user|>" in prompt or "user" in prompt.lower()
        assert "You are helpful" in prompt
        assert "Hello" in prompt

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement format_prompt")
    def test_format_prompt_alpaca(self):
        """Test format_prompt() with alpaca template."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "What is Python?")

        prompt = manager.format_prompt(template="alpaca")

        assert "Instruction" in prompt or "instruction" in prompt.lower()
        assert "What is Python?" in prompt

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement format_prompt")
    def test_format_prompt_vicuna(self):
        """Test format_prompt() with vicuna template."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Tell me a joke")

        prompt = manager.format_prompt(template="vicuna")

        assert "USER:" in prompt
        assert "Tell me a joke" in prompt

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement format_prompt")
    def test_format_prompt_preserves_order(self):
        """Test format_prompt() preserves message order."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("system", "System msg")
        manager.add_message("user", "User msg")
        manager.add_message("assistant", "Assistant msg")

        prompt = manager.format_prompt()

        # Order should be preserved in output
        system_pos = prompt.index("System msg")
        user_pos = prompt.index("User msg")
        assistant_pos = prompt.index("Assistant msg")

        assert system_pos < user_pos < assistant_pos

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement format_prompt")
    def test_format_prompt_default_template(self):
        """Test format_prompt() uses default template when not specified."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Test")

        # Should work without template argument
        prompt = manager.format_prompt()
        assert prompt is not None
        assert "Test" in prompt


# =============================================================================
# Context Window Management Tests
# =============================================================================

class TestContextWindowManagement:
    """Test suite for context window management."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement context management")
    def test_context_window_enforcement(self):
        """Test conversation respects max_context_tokens."""
        from src.conversation import ConversationManager

        manager = ConversationManager(max_context_tokens=100)

        # Add messages that exceed context
        for i in range(20):
            manager.add_message("user", f"Message {i} " * 10)

        # Should truncate or manage context
        assert manager.get_token_count() <= 100

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement context management")
    def test_context_window_keeps_recent_messages(self):
        """Test context window keeps most recent messages."""
        from src.conversation import ConversationManager

        manager = ConversationManager(max_context_tokens=50)

        # Add many messages
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        history = manager.get_history()

        # Should keep most recent
        assert history[-1].content == "Message 9"

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement context management")
    def test_context_window_preserves_system_message(self):
        """Test context window always preserves system message."""
        from src.conversation import ConversationManager

        manager = ConversationManager(max_context_tokens=50)

        manager.add_message("system", "You are a helpful assistant")

        # Add many user messages
        for i in range(10):
            manager.add_message("user", f"Message {i}")

        history = manager.get_history()

        # System message should still be present
        assert history[0].role == "system"


# =============================================================================
# Persistence Tests
# =============================================================================

class TestConversationPersistence:
    """Test suite for conversation save/load."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement save/load")
    def test_save_conversation(self, temp_history_file):
        """Test save() writes conversation to file."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi there")

        manager.save(str(temp_history_file))

        assert temp_history_file.exists()

        # Verify JSON content
        with open(temp_history_file) as f:
            data = json.load(f)
            assert "messages" in data or isinstance(data, list)

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement save/load")
    def test_load_conversation(self, temp_history_file, sample_conversation_data):
        """Test load() reads conversation from file."""
        from src.conversation import ConversationManager

        # Create file with sample data
        with open(temp_history_file, 'w') as f:
            json.dump(sample_conversation_data, f)

        # Load conversation
        manager = ConversationManager.load(str(temp_history_file))

        history = manager.get_history()
        assert len(history) == 3
        assert history[0].role == "system"
        assert history[1].role == "user"
        assert history[2].role == "assistant"

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement save/load")
    def test_save_load_roundtrip(self, temp_history_file):
        """Test conversation survives save/load roundtrip."""
        from src.conversation import ConversationManager

        # Create conversation
        manager1 = ConversationManager()
        manager1.add_message("system", "You are helpful")
        manager1.add_message("user", "What is AI?")
        manager1.add_message("assistant", "AI is artificial intelligence")

        # Save
        manager1.save(str(temp_history_file))

        # Load
        manager2 = ConversationManager.load(str(temp_history_file))

        # Compare
        history1 = manager1.get_history()
        history2 = manager2.get_history()

        assert len(history1) == len(history2)
        for msg1, msg2 in zip(history1, history2):
            assert msg1.role == msg2.role
            assert msg1.content == msg2.content

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement save/load")
    def test_load_nonexistent_file_raises_error(self):
        """Test load() raises error for missing file."""
        from src.conversation import ConversationManager

        with pytest.raises(FileNotFoundError):
            ConversationManager.load("nonexistent_file.json")

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement save/load")
    def test_load_invalid_json_raises_error(self, temp_history_file):
        """Test load() raises error for invalid JSON."""
        from src.conversation import ConversationManager

        # Write invalid JSON
        with open(temp_history_file, 'w') as f:
            f.write("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            ConversationManager.load(str(temp_history_file))


# =============================================================================
# Token Counting Tests
# =============================================================================

class TestTokenCounting:
    """Test suite for token counting."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement token counting")
    def test_token_counting_accuracy(self):
        """Test token counting is reasonably accurate."""
        from src.conversation import ConversationManager

        manager = ConversationManager()

        # Short message
        msg = manager.add_message("user", "Hi")
        assert msg.tokens >= 1

        # Longer message
        msg = manager.add_message("user", "This is a much longer message with many words")
        assert msg.tokens > 5

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement token counting")
    def test_get_token_count_sums_messages(self):
        """Test get_token_count() sums all message tokens."""
        from src.conversation import ConversationManager

        manager = ConversationManager()

        msg1 = manager.add_message("user", "Hello")
        msg2 = manager.add_message("assistant", "Hi there")
        msg3 = manager.add_message("user", "How are you?")

        total = manager.get_token_count()
        expected = msg1.tokens + msg2.tokens + msg3.tokens

        assert total == expected


# =============================================================================
# Prompt Templates Tests
# =============================================================================

class TestPromptTemplates:
    """Test suite for prompt template system."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement templates")
    def test_template_registry_exists(self):
        """Test prompt templates are defined."""
        from src.prompts import TEMPLATES

        assert "chatml" in TEMPLATES
        assert "alpaca" in TEMPLATES
        assert "vicuna" in TEMPLATES

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement templates")
    def test_template_has_placeholders(self):
        """Test templates include necessary placeholders."""
        from src.prompts import TEMPLATES

        # chatml should have role markers
        chatml = TEMPLATES["chatml"]
        assert "{system}" in chatml or "{user}" in chatml or "<|" in chatml

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement templates")
    def test_invalid_template_raises_error(self):
        """Test using invalid template raises error."""
        from src.conversation import ConversationManager

        manager = ConversationManager()
        manager.add_message("user", "Test")

        with pytest.raises((ValueError, KeyError)):
            manager.format_prompt(template="invalid_template")


# =============================================================================
# Integration Tests
# =============================================================================

class TestConversationIntegration:
    """Integration tests for conversation system."""

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement full system")
    def test_full_conversation_workflow(self):
        """Test complete conversation workflow."""
        from src.conversation import ConversationManager

        # Create manager
        manager = ConversationManager(max_context_tokens=2048)

        # Add system message
        manager.add_message("system", "You are a helpful AI assistant")

        # Simulate conversation
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language")
        manager.add_message("user", "Tell me more")
        manager.add_message("assistant", "Python is high-level and interpreted")

        # Verify history
        history = manager.get_history()
        assert len(history) == 5

        # Format for model
        prompt = manager.format_prompt("chatml")
        assert prompt is not None
        assert "Python" in prompt

        # Get stats
        tokens = manager.get_token_count()
        assert tokens > 0
        assert tokens <= 2048

    @pytest.mark.skip(reason="Waiting for conversation-agent to implement full system")
    def test_conversation_with_persistence(self, temp_history_file):
        """Test conversation with save and load."""
        from src.conversation import ConversationManager

        # Create and populate conversation
        manager1 = ConversationManager()
        manager1.add_message("system", "You are helpful")
        manager1.add_message("user", "Hello")
        manager1.add_message("assistant", "Hi there!")

        # Save
        manager1.save(str(temp_history_file))

        # Load in new manager
        manager2 = ConversationManager.load(str(temp_history_file))

        # Continue conversation
        manager2.add_message("user", "How are you?")

        # Verify continuity
        history = manager2.get_history()
        assert len(history) == 4
        assert history[0].content == "You are helpful"
        assert history[-1].content == "How are you?"
