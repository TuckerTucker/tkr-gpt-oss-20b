"""
Unit tests for HarmonyPromptBuilder.

This module tests the HarmonyPromptBuilder class that builds Harmony-compliant
prompts using the official openai-harmony package.

Coverage requirement: >95%
"""

import pytest
import time
import sys
import os
from typing import List, Dict

# Add the contract path to allow imports
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-replacement', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_builder_interface import (
    HarmonyPrompt,
    ReasoningLevel,
    PERFORMANCE_CONTRACT,
)

from src.prompts.harmony_native import HarmonyPromptBuilder


class TestSystemPromptBuilding:
    """Test system prompt building with various configurations."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_system_prompt_basic(self, builder):
        """Test basic system prompt building."""
        result = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert isinstance(result, HarmonyPrompt)
        assert isinstance(result.token_ids, list)
        assert len(result.token_ids) > 0
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert isinstance(result.metadata, dict)

        # Verify metadata
        assert result.metadata["reasoning_level"] == "medium"
        assert result.metadata["knowledge_cutoff"] == "2024-06"
        assert result.metadata["current_date"] == "2025-10-27"
        assert result.metadata["include_function_tools"] is False

    def test_system_prompt_all_reasoning_levels(self, builder):
        """Test system prompt with all reasoning levels."""
        for level in [ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]:
            result = builder.build_system_prompt(
                reasoning_level=level,
                knowledge_cutoff="2024-06",
                current_date="2025-10-27"
            )

            assert isinstance(result, HarmonyPrompt)
            assert len(result.token_ids) > 0
            assert result.metadata["reasoning_level"] == level.value

    def test_system_prompt_with_function_tools(self, builder):
        """Test system prompt with function tool routing enabled."""
        result = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["include_function_tools"] is True

    def test_system_prompt_validates_inputs(self, builder):
        """Test that system prompt validates inputs."""
        # Empty knowledge_cutoff
        with pytest.raises(ValueError, match="knowledge_cutoff cannot be empty"):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="",
                current_date="2025-10-27"
            )

        # Empty current_date
        with pytest.raises(ValueError, match="current_date cannot be empty"):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="2024-06",
                current_date=""
            )

    def test_system_prompt_token_ids_are_integers(self, builder):
        """Test that token IDs are valid integers."""
        result = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert all(isinstance(token_id, int) for token_id in result.token_ids)
        assert all(token_id >= 0 for token_id in result.token_ids)

    def test_system_prompt_deterministic(self, builder):
        """Test that same inputs produce same outputs (idempotency)."""
        result1 = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        result2 = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert result1.token_ids == result2.token_ids
        assert result1.text == result2.text


class TestDeveloperPromptBuilding:
    """Test developer prompt building with various configurations."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_developer_prompt_basic(self, builder):
        """Test basic developer prompt building."""
        instructions = "You are a helpful assistant that provides clear, concise answers."

        result = builder.build_developer_prompt(instructions=instructions)

        assert isinstance(result, HarmonyPrompt)
        assert isinstance(result.token_ids, list)
        assert len(result.token_ids) > 0
        assert isinstance(result.text, str)
        assert "# Instructions" in result.text
        assert instructions in result.text

        # Verify metadata
        assert result.metadata["has_instructions"] is True
        assert result.metadata["has_tools"] is False
        assert result.metadata["tool_count"] == 0

    def test_developer_prompt_with_tools(self, builder):
        """Test developer prompt with function tools."""
        instructions = "You can use tools to help the user."
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

        result = builder.build_developer_prompt(
            instructions=instructions,
            function_tools=tools
        )

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert "# Instructions" in result.text
        assert "# Tools" in result.text
        assert "get_weather" in result.text

        # Verify metadata
        assert result.metadata["has_tools"] is True
        assert result.metadata["tool_count"] == 1

    def test_developer_prompt_with_multiple_tools(self, builder):
        """Test developer prompt with multiple function tools."""
        instructions = "Use the available tools to help the user."
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        ]

        result = builder.build_developer_prompt(
            instructions=instructions,
            function_tools=tools
        )

        assert isinstance(result, HarmonyPrompt)
        assert "search" in result.text
        assert "calculate" in result.text
        assert result.metadata["tool_count"] == 2

    def test_developer_prompt_validates_empty_instructions(self, builder):
        """Test that developer prompt validates empty instructions."""
        with pytest.raises(ValueError, match="instructions cannot be empty"):
            builder.build_developer_prompt(instructions="")

        with pytest.raises(ValueError, match="instructions cannot be empty"):
            builder.build_developer_prompt(instructions="   ")

    def test_developer_prompt_validates_tools(self, builder):
        """Test that developer prompt validates tool structure."""
        instructions = "Test instructions"

        # Tools not a list
        with pytest.raises(ValueError, match="function_tools must be a list"):
            builder.build_developer_prompt(
                instructions=instructions,
                function_tools="not a list"
            )

        # Tool not a dict
        with pytest.raises(ValueError, match="must be a dict"):
            builder.build_developer_prompt(
                instructions=instructions,
                function_tools=["not a dict"]
            )

        # Tool missing name
        with pytest.raises(ValueError, match="missing 'name' field"):
            builder.build_developer_prompt(
                instructions=instructions,
                function_tools=[{"description": "test"}]
            )

    def test_developer_prompt_token_ids_are_integers(self, builder):
        """Test that token IDs are valid integers."""
        result = builder.build_developer_prompt(
            instructions="Test instructions"
        )

        assert all(isinstance(token_id, int) for token_id in result.token_ids)
        assert all(token_id >= 0 for token_id in result.token_ids)

    def test_developer_prompt_deterministic(self, builder):
        """Test that same inputs produce same outputs."""
        instructions = "Test instructions"
        tools = [{"name": "test_tool", "description": "A test tool"}]

        result1 = builder.build_developer_prompt(
            instructions=instructions,
            function_tools=tools
        )

        result2 = builder.build_developer_prompt(
            instructions=instructions,
            function_tools=tools
        )

        assert result1.token_ids == result2.token_ids
        assert result1.text == result2.text


class TestConversationBuilding:
    """Test conversation building with various configurations."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_conversation_single_turn(self, builder):
        """Test building a single-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        result = builder.build_conversation(messages=messages)

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["message_count"] == 1
        assert result.metadata["has_system_prompt"] is False
        assert result.metadata["has_developer_prompt"] is False
        assert result.metadata["include_generation_prompt"] is True

    def test_conversation_multi_turn(self, builder):
        """Test building a multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I can help with that!"}
        ]

        result = builder.build_conversation(messages=messages)

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["message_count"] == 4

    def test_conversation_with_system_prompt(self, builder):
        """Test building conversation with system prompt."""
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        result = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt
        )

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["has_system_prompt"] is True

    def test_conversation_with_developer_prompt(self, builder):
        """Test building conversation with developer prompt."""
        developer_prompt = builder.build_developer_prompt(
            instructions="Be helpful and concise."
        )

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        result = builder.build_conversation(
            messages=messages,
            developer_prompt=developer_prompt
        )

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["has_developer_prompt"] is True

    def test_conversation_complete(self, builder):
        """Test building complete conversation with all components."""
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        developer_prompt = builder.build_developer_prompt(
            instructions="You are a helpful assistant.",
            function_tools=[
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            ]
        )

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        result = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            include_generation_prompt=True
        )

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0
        assert result.metadata["message_count"] == 3
        assert result.metadata["has_system_prompt"] is True
        assert result.metadata["has_developer_prompt"] is True
        assert result.metadata["include_generation_prompt"] is True

    def test_conversation_without_generation_prompt(self, builder):
        """Test building conversation without generation prompt."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"}
        ]

        result = builder.build_conversation(
            messages=messages,
            include_generation_prompt=False
        )

        assert isinstance(result, HarmonyPrompt)
        assert result.metadata["include_generation_prompt"] is False

    def test_conversation_validates_messages(self, builder):
        """Test that conversation validates message structure."""
        # Messages not a list
        with pytest.raises(ValueError, match="messages must be a list"):
            builder.build_conversation(messages="not a list")

        # Message not a dict
        with pytest.raises(ValueError, match="must be a dict"):
            builder.build_conversation(messages=["not a dict"])

        # Message missing role
        with pytest.raises(ValueError, match="missing 'role' field"):
            builder.build_conversation(messages=[{"content": "test"}])

        # Message missing content
        with pytest.raises(ValueError, match="missing 'content' field"):
            builder.build_conversation(messages=[{"role": "user"}])

        # Invalid role
        with pytest.raises(ValueError, match="Invalid role"):
            builder.build_conversation(
                messages=[{"role": "invalid", "content": "test"}]
            )

    def test_conversation_token_ids_are_integers(self, builder):
        """Test that token IDs are valid integers."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        result = builder.build_conversation(messages=messages)

        assert all(isinstance(token_id, int) for token_id in result.token_ids)
        assert all(token_id >= 0 for token_id in result.token_ids)

    def test_conversation_with_channels(self, builder):
        """Test conversation with channel annotations."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!", "channel": "final"}
        ]

        result = builder.build_conversation(messages=messages)

        assert isinstance(result, HarmonyPrompt)
        assert len(result.token_ids) > 0

    def test_conversation_deterministic(self, builder):
        """Test that same inputs produce same outputs."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"}
        ]

        result1 = builder.build_conversation(messages=messages)
        result2 = builder.build_conversation(messages=messages)

        assert result1.token_ids == result2.token_ids


class TestPerformanceContract:
    """Test that performance requirements are met."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_system_prompt_performance(self, builder):
        """Test that system prompt building meets performance contract."""
        max_latency_ms = PERFORMANCE_CONTRACT["build_system_prompt"]["max_latency_ms"]

        # Warm up (JIT compilation, etc.)
        for _ in range(3):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="2024-06",
                current_date="2025-10-27"
            )

        # Measure performance
        iterations = 10
        total_time = 0

        for _ in range(iterations):
            start = time.perf_counter()
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="2024-06",
                current_date="2025-10-27"
            )
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            total_time += elapsed

        avg_time_ms = total_time / iterations

        # Assert performance meets contract
        assert avg_time_ms < max_latency_ms, (
            f"System prompt building took {avg_time_ms:.2f}ms on average, "
            f"exceeds contract of {max_latency_ms}ms"
        )

    def test_developer_prompt_performance(self, builder):
        """Test that developer prompt building meets performance contract."""
        max_latency_ms = PERFORMANCE_CONTRACT["build_developer_prompt"]["max_latency_ms"]

        instructions = "You are a helpful assistant."
        tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "number"}
                    }
                }
            }
        ]

        # Warm up
        for _ in range(3):
            builder.build_developer_prompt(
                instructions=instructions,
                function_tools=tools
            )

        # Measure performance
        iterations = 10
        total_time = 0

        for _ in range(iterations):
            start = time.perf_counter()
            builder.build_developer_prompt(
                instructions=instructions,
                function_tools=tools
            )
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed

        avg_time_ms = total_time / iterations

        assert avg_time_ms < max_latency_ms, (
            f"Developer prompt building took {avg_time_ms:.2f}ms on average, "
            f"exceeds contract of {max_latency_ms}ms"
        )

    def test_conversation_performance(self, builder):
        """Test that conversation building meets performance contract."""
        max_latency_per_message_ms = PERFORMANCE_CONTRACT["build_conversation"]["max_latency_ms"]

        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]

        # Warm up
        for _ in range(3):
            builder.build_conversation(messages=messages)

        # Measure performance
        iterations = 10
        total_time = 0

        for _ in range(iterations):
            start = time.perf_counter()
            builder.build_conversation(messages=messages)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed

        avg_time_ms = total_time / iterations
        avg_time_per_message_ms = avg_time_ms / len(messages)

        assert avg_time_per_message_ms < max_latency_per_message_ms, (
            f"Conversation building took {avg_time_per_message_ms:.2f}ms per message, "
            f"exceeds contract of {max_latency_per_message_ms}ms per message"
        )


class TestThreadSafety:
    """Test thread safety requirements."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_builder_is_stateless(self, builder):
        """Test that builder maintains no mutable state between calls."""
        # Build first prompt
        result1 = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.LOW,
            knowledge_cutoff="2024-01",
            current_date="2025-01-01"
        )

        # Build second prompt with different params
        result2 = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        # Build first again - should be identical to result1
        result3 = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.LOW,
            knowledge_cutoff="2024-01",
            current_date="2025-01-01"
        )

        assert result1.token_ids == result3.token_ids
        assert result1.token_ids != result2.token_ids


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def builder(self):
        """Create a HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    def test_complete_workflow(self, builder):
        """Test complete workflow from prompts to conversation."""
        # Build system prompt
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        assert len(system_prompt.token_ids) > 0

        # Build developer prompt
        developer_prompt = builder.build_developer_prompt(
            instructions="You are a helpful AI assistant. Be concise and accurate.",
            function_tools=[
                {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            ]
        )

        assert len(developer_prompt.token_ids) > 0

        # Build conversation
        messages = [
            {"role": "user", "content": "Can you search for information about Python?"},
            {"role": "assistant", "content": "I'll search for that.", "channel": "final"}
        ]

        conversation = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            include_generation_prompt=True
        )

        assert len(conversation.token_ids) > 0
        assert conversation.metadata["message_count"] == 2
        assert conversation.metadata["has_system_prompt"] is True
        assert conversation.metadata["has_developer_prompt"] is True

    def test_mlx_token_format_compatibility(self, builder):
        """Test that token format is compatible with MLX."""
        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        result = builder.build_conversation(messages=messages)

        # Token IDs should be a list of integers (MLX-compatible)
        assert isinstance(result.token_ids, list)
        assert all(isinstance(tid, int) for tid in result.token_ids)
        assert all(tid >= 0 for tid in result.token_ids)

        # Should be able to convert to various formats MLX might need
        import array
        token_array = array.array('i', result.token_ids)  # Integer array
        assert len(token_array) == len(result.token_ids)
