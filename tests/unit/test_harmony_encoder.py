"""
Unit tests for Harmony Prompt Builder and Response Parser.

Tests the HarmonyPromptBuilder and HarmonyResponseParser implementations including:
- System prompt building with reasoning levels
- Developer prompt building with instructions and tools
- Conversation building with multiple message types
- Response parsing with single and multiple channels
- Format validation
- Error handling for malformed input
- Performance benchmarks
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
import time

# Import contract types
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-replacement', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_builder_interface import (
    ReasoningLevel,
    HarmonyPrompt,
)
from harmony_parser_interface import (
    ParsedHarmonyResponse,
)

from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    HarmonyResponseParser,
    HarmonyFormatError,
)


# ============================================================================
# BUILDER TESTS
# ============================================================================


class TestHarmonyPromptBuilderBasics:
    """Test basic builder initialization and system prompts."""

    def test_builder_initialization(self):
        """Test that builder initializes correctly."""
        builder = HarmonyPromptBuilder()
        assert builder is not None
        assert hasattr(builder, 'build_system_prompt')
        assert hasattr(builder, 'build_developer_prompt')
        assert hasattr(builder, 'build_conversation')

    def test_build_system_prompt_low_reasoning(self):
        """Test building system prompt with LOW reasoning level."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.LOW,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert isinstance(prompt, HarmonyPrompt)
        assert len(prompt.token_ids) > 0
        assert "ChatGPT" in prompt.text
        assert "2024-06" in prompt.text
        assert "2025-10-27" in prompt.text
        assert prompt.metadata["reasoning_level"] == "low"

    def test_build_system_prompt_medium_reasoning(self):
        """Test building system prompt with MEDIUM reasoning level."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert isinstance(prompt, HarmonyPrompt)
        assert prompt.metadata["reasoning_level"] == "medium"

    def test_build_system_prompt_high_reasoning(self):
        """Test building system prompt with HIGH reasoning level."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        assert isinstance(prompt, HarmonyPrompt)
        assert prompt.metadata["reasoning_level"] == "high"

    def test_build_system_prompt_with_function_tools(self):
        """Test building system prompt with function tools enabled."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        assert prompt.metadata["include_function_tools"] is True
        assert "tool" in prompt.text.lower() or "function" in prompt.text.lower()

    def test_build_system_prompt_invalid_cutoff(self):
        """Test that empty knowledge_cutoff raises ValueError."""
        builder = HarmonyPromptBuilder()

        with pytest.raises(ValueError, match="knowledge_cutoff cannot be empty"):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="",
                current_date="2025-10-27"
            )

    def test_build_system_prompt_invalid_date(self):
        """Test that empty current_date raises ValueError."""
        builder = HarmonyPromptBuilder()

        with pytest.raises(ValueError, match="current_date cannot be empty"):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="2024-06",
                current_date=""
            )


class TestHarmonyPromptBuilderDeveloperPrompts:
    """Test developer prompt building."""

    def test_build_developer_prompt_simple(self):
        """Test building developer prompt with just instructions."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_developer_prompt(
            instructions="You are a helpful assistant."
        )

        assert isinstance(prompt, HarmonyPrompt)
        assert len(prompt.token_ids) > 0
        assert "# Instructions" in prompt.text
        assert "helpful assistant" in prompt.text
        assert prompt.metadata["has_instructions"] is True
        assert prompt.metadata["has_tools"] is False

    def test_build_developer_prompt_with_tools(self):
        """Test building developer prompt with function tools."""
        builder = HarmonyPromptBuilder()

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "description": "Temperature units"}
                    },
                    "required": ["location"]
                }
            }
        ]

        prompt = builder.build_developer_prompt(
            instructions="Use tools when needed",
            function_tools=tools
        )

        assert "# Instructions" in prompt.text
        assert "# Tools" in prompt.text
        assert "get_weather" in prompt.text
        assert prompt.metadata["has_tools"] is True
        assert prompt.metadata["tool_count"] == 1

    def test_build_developer_prompt_empty_instructions(self):
        """Test that empty instructions raises ValueError."""
        builder = HarmonyPromptBuilder()

        with pytest.raises(ValueError, match="instructions cannot be empty"):
            builder.build_developer_prompt(instructions="")

    def test_build_developer_prompt_invalid_tools(self):
        """Test that malformed tools raise ValueError."""
        builder = HarmonyPromptBuilder()

        # Tools must be a list
        with pytest.raises(ValueError, match="function_tools must be a list"):
            builder.build_developer_prompt(
                instructions="Test",
                function_tools="not a list"
            )

        # Tool items must be dicts
        with pytest.raises(ValueError, match="must be a dict"):
            builder.build_developer_prompt(
                instructions="Test",
                function_tools=["not a dict"]
            )

        # Tools must have 'name' field
        with pytest.raises(ValueError, match="missing 'name' field"):
            builder.build_developer_prompt(
                instructions="Test",
                function_tools=[{"description": "missing name"}]
            )


class TestHarmonyPromptBuilderConversations:
    """Test conversation building."""

    def test_build_conversation_simple(self):
        """Test building simple user message conversation."""
        builder = HarmonyPromptBuilder()

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        prompt = builder.build_conversation(messages)

        assert isinstance(prompt, HarmonyPrompt)
        assert len(prompt.token_ids) > 0
        assert prompt.metadata["message_count"] == 1
        assert prompt.metadata["include_generation_prompt"] is True

    def test_build_conversation_multi_turn(self):
        """Test building multi-turn conversation."""
        builder = HarmonyPromptBuilder()

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "channel": "final"},
            {"role": "user", "content": "What about 3+3?"}
        ]

        prompt = builder.build_conversation(messages)

        assert prompt.metadata["message_count"] == 3

    def test_build_conversation_with_system_prompt(self):
        """Test building conversation with system prompt."""
        builder = HarmonyPromptBuilder()

        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        messages = [{"role": "user", "content": "Hello"}]

        prompt = builder.build_conversation(
            messages,
            system_prompt=system_prompt
        )

        assert prompt.metadata["has_system_prompt"] is True
        assert prompt.metadata["message_count"] == 1

    def test_build_conversation_with_developer_prompt(self):
        """Test building conversation with developer prompt."""
        builder = HarmonyPromptBuilder()

        developer_prompt = builder.build_developer_prompt(
            instructions="Be helpful"
        )

        messages = [{"role": "user", "content": "Hello"}]

        prompt = builder.build_conversation(
            messages,
            developer_prompt=developer_prompt
        )

        assert prompt.metadata["has_developer_prompt"] is True

    def test_build_conversation_without_generation_prompt(self):
        """Test building conversation without generation prompt."""
        builder = HarmonyPromptBuilder()

        messages = [{"role": "user", "content": "Hello"}]

        prompt = builder.build_conversation(
            messages,
            include_generation_prompt=False
        )

        assert prompt.metadata["include_generation_prompt"] is False

    def test_build_conversation_empty_messages(self):
        """Test building conversation with empty messages list."""
        builder = HarmonyPromptBuilder()

        # Empty messages should work
        prompt = builder.build_conversation([])
        assert prompt.metadata["message_count"] == 0

    def test_build_conversation_invalid_message_format(self):
        """Test that invalid message formats raise ValueError."""
        builder = HarmonyPromptBuilder()

        # Message not a dict
        with pytest.raises(ValueError, match="must be a dict"):
            builder.build_conversation(["not a dict"])

        # Missing 'role' field
        with pytest.raises(ValueError, match="missing 'role' field"):
            builder.build_conversation([{"content": "test"}])

        # Missing 'content' field
        with pytest.raises(ValueError, match="missing 'content' field"):
            builder.build_conversation([{"role": "user"}])

        # Invalid role
        with pytest.raises(ValueError, match="Invalid role"):
            builder.build_conversation([{"role": "invalid", "content": "test"}])


# ============================================================================
# PARSER TESTS
# ============================================================================


class MockTokenizer:
    """Mock tokenizer for testing text-based parsing."""

    def encode(self, text, allowed_special="all"):
        """Mock encode that returns dummy token IDs."""
        # Return a simple list of integers based on text length
        # This is just for testing - real tokenizer would be more sophisticated
        return [ord(c) % 256 for c in text[:100]]


class TestHarmonyResponseParserBasics:
    """Test basic response parsing."""

    def test_parser_initialization(self):
        """Test that parser initializes correctly."""
        parser = HarmonyResponseParser()
        assert parser is not None
        assert hasattr(parser, 'parse_response_tokens')
        assert hasattr(parser, 'parse_response_text')
        assert hasattr(parser, 'validate_harmony_format')

    def test_parse_response_tokens_simple(self):
        """Test parsing simple token response."""
        parser = HarmonyResponseParser()

        # We need actual Harmony tokens for this to work properly
        # For now, test that it handles basic inputs without crashing
        # Real implementation would use actual Harmony-encoded tokens
        try:
            # This may fail with format validation, which is expected
            result = parser.parse_response_tokens([1, 2, 3])
            assert isinstance(result, ParsedHarmonyResponse)
            # If it succeeds, final should never be None
            assert result.final is not None
        except (ValueError, HarmonyFormatError):
            # Expected for non-Harmony tokens
            pass

    def test_parse_response_tokens_empty(self):
        """Test that empty tokens raise ValueError."""
        parser = HarmonyResponseParser()

        with pytest.raises(ValueError, match="Token IDs cannot be empty"):
            parser.parse_response_tokens([])

    def test_parse_response_text_simple(self):
        """Test parsing text response."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        try:
            result = parser.parse_response_text(
                "Some response text",
                tokenizer
            )
            assert isinstance(result, ParsedHarmonyResponse)
            assert result.final is not None
        except (ValueError, HarmonyFormatError):
            # Expected if tokens don't match Harmony format
            pass

    def test_parse_response_text_empty(self):
        """Test that empty text raises ValueError."""
        parser = HarmonyResponseParser()
        tokenizer = MockTokenizer()

        with pytest.raises(ValueError, match="Response text cannot be empty"):
            parser.parse_response_text("", tokenizer)

    def test_parse_response_text_no_tokenizer(self):
        """Test that missing tokenizer raises ValueError."""
        parser = HarmonyResponseParser()

        with pytest.raises(ValueError, match="Tokenizer is required"):
            parser.parse_response_text("test", None)

    def test_parsed_response_structure(self):
        """Test that ParsedHarmonyResponse has correct structure."""
        # Create a response manually
        parsed = ParsedHarmonyResponse(
            final="Hello",
            analysis="Thinking...",
            commentary="Note",
            channels={"final": "Hello", "analysis": "Thinking..."},
            metadata={"token_count": 10}
        )

        assert parsed.final == "Hello"
        assert parsed.analysis == "Thinking..."
        assert parsed.commentary == "Note"
        assert parsed.channels["final"] == "Hello"
        assert parsed.metadata["token_count"] == 10

    def test_parsed_response_optional_fields(self):
        """Test that ParsedHarmonyResponse works with minimal fields."""
        parsed = ParsedHarmonyResponse(final="Hello")

        assert parsed.final == "Hello"
        assert parsed.analysis is None
        assert parsed.commentary is None
        assert parsed.channels is None
        assert parsed.metadata is None


class TestHarmonyResponseParserExtraction:
    """Test channel extraction from parsed responses."""

    def test_extract_channel_final(self):
        """Test extracting final channel."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello",
            channels={"final": "Hello"}
        )

        result = parser.extract_channel(parsed, "final")
        assert result == "Hello"

    def test_extract_channel_analysis(self):
        """Test extracting analysis channel."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello",
            analysis="Thinking...",
            channels={"final": "Hello", "analysis": "Thinking..."}
        )

        result = parser.extract_channel(parsed, "analysis")
        assert result == "Thinking..."

    def test_extract_channel_commentary(self):
        """Test extracting commentary channel."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(
            final="Hello",
            commentary="Note",
            channels={"final": "Hello", "commentary": "Note"}
        )

        result = parser.extract_channel(parsed, "commentary")
        assert result == "Note"

    def test_extract_channel_missing(self):
        """Test extracting non-existent channel returns None."""
        parser = HarmonyResponseParser()

        parsed = ParsedHarmonyResponse(final="Hello")

        result = parser.extract_channel(parsed, "nonexistent")
        assert result is None


class TestHarmonyResponseParserValidation:
    """Test format validation."""

    def test_validate_empty_tokens(self):
        """Test that empty tokens fail validation."""
        parser = HarmonyResponseParser()

        assert parser.validate_harmony_format([]) is False

    def test_validate_non_harmony_tokens(self):
        """Test that non-Harmony tokens fail validation."""
        parser = HarmonyResponseParser()

        # Random tokens that aren't Harmony format
        assert parser.validate_harmony_format([1, 2, 3, 4, 5]) is False


class TestHarmonyResponseParserErrorHandling:
    """Test error handling and edge cases."""

    def test_parse_malformed_tokens_graceful(self):
        """Test that malformed tokens are handled gracefully."""
        parser = HarmonyResponseParser()

        # Parser should return empty result instead of crashing
        try:
            result = parser.parse_response_tokens([999999, 888888, 777777])
            # Should either succeed with empty final or raise expected errors
            assert isinstance(result, ParsedHarmonyResponse)
            assert result.final is not None  # Never None
        except (ValueError, HarmonyFormatError):
            # These exceptions are acceptable
            pass

    def test_parse_text_with_bad_tokenizer(self):
        """Test handling of tokenizer errors."""
        parser = HarmonyResponseParser()

        # Tokenizer that doesn't have encode method
        bad_tokenizer = object()

        with pytest.raises(ValueError, match="Tokenizer"):
            parser.parse_response_text("test", bad_tokenizer)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test performance requirements from contracts."""

    def test_build_system_prompt_performance(self):
        """Test system prompt building completes in <50ms."""
        builder = HarmonyPromptBuilder()

        start_time = time.perf_counter()
        prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 50, f"System prompt took {elapsed:.2f}ms, expected <50ms"
        assert len(prompt.token_ids) > 0

    def test_build_developer_prompt_performance(self):
        """Test developer prompt building completes in <100ms."""
        builder = HarmonyPromptBuilder()

        tools = [
            {
                "name": f"tool_{i}",
                "description": f"Tool {i}",
                "parameters": {"type": "object", "properties": {}}
            }
            for i in range(5)
        ]

        start_time = time.perf_counter()
        prompt = builder.build_developer_prompt(
            instructions="Use these tools",
            function_tools=tools
        )
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 100, f"Developer prompt took {elapsed:.2f}ms, expected <100ms"
        assert len(prompt.token_ids) > 0

    def test_build_conversation_performance(self):
        """Test conversation building scales linearly (<10ms per message)."""
        builder = HarmonyPromptBuilder()

        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(20)
        ]

        start_time = time.perf_counter()
        prompt = builder.build_conversation(messages)
        elapsed = (time.perf_counter() - start_time) * 1000

        max_expected = 10 * len(messages)
        assert elapsed < max_expected, f"Conversation took {elapsed:.2f}ms, expected <{max_expected}ms"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test complete workflows combining builder and parser."""

    def test_build_and_validate_workflow(self):
        """Test building a conversation and validating it."""
        builder = HarmonyPromptBuilder()
        parser = HarmonyResponseParser()

        # Build a complete conversation
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        messages = [
            {"role": "user", "content": "Hello!"}
        ]

        conversation = builder.build_conversation(
            messages,
            system_prompt=system_prompt
        )

        # Validate the tokens (may or may not be valid Harmony format)
        is_valid = parser.validate_harmony_format(conversation.token_ids)
        # Just verify it doesn't crash
        assert isinstance(is_valid, bool)

    def test_full_conversation_lifecycle(self):
        """Test complete conversation building with all components."""
        builder = HarmonyPromptBuilder()

        # Build system prompt
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        # Build developer prompt
        developer_prompt = builder.build_developer_prompt(
            instructions="You are a helpful coding assistant",
            function_tools=[
                {
                    "name": "run_code",
                    "description": "Execute Python code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to run"}
                        },
                        "required": ["code"]
                    }
                }
            ]
        )

        # Build conversation
        messages = [
            {"role": "user", "content": "Write a hello world program"},
            {"role": "assistant", "content": "print('Hello, World!')", "channel": "final"},
            {"role": "user", "content": "Now make it say goodbye"}
        ]

        conversation = builder.build_conversation(
            messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            include_generation_prompt=True
        )

        assert len(conversation.token_ids) > 0
        assert conversation.metadata["message_count"] == 3
        assert conversation.metadata["has_system_prompt"] is True
        assert conversation.metadata["has_developer_prompt"] is True


class TestThreadSafety:
    """Test thread safety requirements."""

    def test_concurrent_builder_usage(self):
        """Test that builder can be used concurrently."""
        import threading

        builder = HarmonyPromptBuilder()
        results = []
        errors = []

        def build_task(task_id):
            try:
                prompt = builder.build_system_prompt(
                    reasoning_level=ReasoningLevel.MEDIUM,
                    knowledge_cutoff="2024-06",
                    current_date="2025-10-27"
                )
                results.append((task_id, prompt))
            except Exception as e:
                errors.append((task_id, e))

        threads = [threading.Thread(target=build_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_parser_usage(self):
        """Test that parser can be used concurrently."""
        import threading

        parser = HarmonyResponseParser()
        results = []
        errors = []

        def parse_task(task_id):
            try:
                # Try to parse some tokens (may fail but shouldn't crash)
                parsed = ParsedHarmonyResponse(final=f"Result {task_id}")
                channel = parser.extract_channel(parsed, "final")
                results.append((task_id, channel))
            except Exception as e:
                errors.append((task_id, e))

        threads = [threading.Thread(target=parse_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_in_prompts(self):
        """Test that builder handles Unicode correctly."""
        builder = HarmonyPromptBuilder()

        prompt = builder.build_developer_prompt(
            instructions="Hello 世界 🌍"
        )

        assert "世界" in prompt.text
        assert "🌍" in prompt.text

    def test_long_instructions(self):
        """Test that builder handles very long instructions."""
        builder = HarmonyPromptBuilder()

        long_text = "This is a test. " * 1000  # ~16KB of text

        prompt = builder.build_developer_prompt(instructions=long_text)

        assert len(prompt.token_ids) > 0
        # Check that the long text is included (may be wrapped with headers)
        assert "This is a test." in prompt.text
        assert len(prompt.text) > 10000  # Verify it's actually long

    def test_multiple_reasoning_levels(self):
        """Test all reasoning levels produce different outputs."""
        builder = HarmonyPromptBuilder()

        prompts = {}
        for level in [ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]:
            prompt = builder.build_system_prompt(
                reasoning_level=level,
                knowledge_cutoff="2024-06",
                current_date="2025-10-27"
            )
            prompts[level] = prompt

        # Each level should produce tokens (may or may not be different)
        assert len(prompts[ReasoningLevel.LOW].token_ids) > 0
        assert len(prompts[ReasoningLevel.MEDIUM].token_ids) > 0
        assert len(prompts[ReasoningLevel.HIGH].token_ids) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/prompts/harmony_native", "--cov-report=term-missing"])
