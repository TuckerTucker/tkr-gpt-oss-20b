"""
Integration tests for HarmonyPromptBuilder and HarmonyResponseParser.

This module validates that the builder and parser work together correctly
in complete round-trip workflows.

Gate Criteria (ALL MUST PASS for Wave 2):
- All round-trip tests pass
- All contract compliance tests pass
- All cross-component tests pass
- Performance integration tests pass
- Data flow validation tests pass

Minimum: 20+ integration tests, all passing
"""

import pytest
import time
import sys
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add contract path
contract_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    '.context-kit', 'orchestration', 'harmony-replacement', 'integration-contracts'
)
if os.path.exists(contract_path):
    sys.path.insert(0, contract_path)

from harmony_builder_interface import (
    HarmonyPrompt,
    ReasoningLevel,
    HarmonyPromptBuilderInterface,
    PERFORMANCE_CONTRACT as BUILDER_PERF,
)
from harmony_parser_interface import (
    ParsedHarmonyResponse,
    HarmonyResponseParserInterface,
    PERFORMANCE_CONTRACT as PARSER_PERF,
)

from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    HarmonyResponseParser,
)

from openai_harmony import load_harmony_encoding, HarmonyEncodingName


# ============================================================================
# MOCK RESPONSE HELPERS
# ============================================================================

def create_mock_harmony_response(
    analysis: str = None,
    commentary: str = None,
    final: str = "Test response"
) -> str:
    """
    Create mock Harmony-formatted response for testing.

    Args:
        analysis: Optional analysis channel content
        commentary: Optional commentary channel content
        final: Final channel content (required)

    Returns:
        Harmony-formatted response string
    """
    parts = []

    if analysis:
        parts.append(f'<|channel|>analysis<|message|>{analysis}<|end|>')

    if commentary:
        parts.append(f'<|start|>assistant<|channel|>commentary<|message|>{commentary}<|end|>')

    parts.append(f'<|start|>assistant<|channel|>final<|message|>{final}<|end|>')

    return '\n'.join(parts)


def encode_mock_response(response_text: str) -> List[int]:
    """
    Encode mock response to token IDs.

    Args:
        response_text: Harmony-formatted text

    Returns:
        List of token IDs
    """
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return list(encoding.encode(response_text, allowed_special="all"))


# ============================================================================
# A. ROUND-TRIP INTEGRATION TESTS
# ============================================================================

class TestRoundTripIntegration:
    """Test complete build → parse round-trip."""

    @pytest.fixture
    def builder(self):
        """Create HarmonyPromptBuilder instance."""
        return HarmonyPromptBuilder()

    @pytest.fixture
    def parser(self):
        """Create HarmonyResponseParser instance."""
        return HarmonyResponseParser()

    @pytest.fixture
    def encoding(self):
        """Create Harmony encoding for tokenization."""
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def test_system_prompt_round_trip(self, builder, parser, encoding):
        """System prompt: build → render → decode → validate format."""
        # Build system prompt
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        # Verify token IDs are valid integers
        assert isinstance(system_prompt.token_ids, list)
        assert len(system_prompt.token_ids) > 0
        assert all(isinstance(tid, int) for tid in system_prompt.token_ids)

        # Decode tokens back to text
        decoded_text = encoding.decode(system_prompt.token_ids)
        assert isinstance(decoded_text, str)
        assert len(decoded_text) > 0

        # Verify structure matches spec
        assert system_prompt.metadata["reasoning_level"] == "high"
        assert system_prompt.metadata["knowledge_cutoff"] == "2024-06"
        assert system_prompt.metadata["current_date"] == "2025-10-27"
        assert system_prompt.metadata["include_function_tools"] is True

    def test_conversation_round_trip(self, builder, parser, encoding):
        """Full conversation: build → mock response → parse → validate."""
        # Build conversation
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me calculate that.", "channel": "final"},
            {"role": "user", "content": "Please show your work."}
        ]

        conversation = builder.build_conversation(
            messages=messages,
            include_generation_prompt=True
        )

        # Verify conversation built correctly
        assert len(conversation.token_ids) > 0
        assert conversation.metadata["message_count"] == 3

        # Simulate model response
        mock_response = create_mock_harmony_response(
            analysis="2 + 2 = 4 (basic arithmetic)",
            commentary="This is a simple math question",
            final="2 + 2 = 4"
        )
        response_tokens = encode_mock_response(mock_response)

        # Parse response
        parsed = parser.parse_response_tokens(response_tokens)

        # Validate channels extracted correctly
        assert parsed.final == "2 + 2 = 4"
        assert parsed.analysis == "2 + 2 = 4 (basic arithmetic)"
        assert parsed.commentary == "This is a simple math question"
        assert parsed.channels is not None
        assert "final" in parsed.channels
        assert "analysis" in parsed.channels
        assert "commentary" in parsed.channels

    def test_multi_turn_round_trip(self, builder, parser, encoding):
        """Multi-turn conversation round-trip."""
        # Build system prompt
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        # Build developer prompt
        developer_prompt = builder.build_developer_prompt(
            instructions="You are a helpful math tutor."
        )

        # Build multi-turn conversation
        messages = [
            {"role": "user", "content": "What is 5 * 6?"},
            {"role": "assistant", "content": "5 * 6 = 30", "channel": "final"},
            {"role": "user", "content": "What about 7 * 8?"},
            {"role": "assistant", "content": "7 * 8 = 56", "channel": "final"},
            {"role": "user", "content": "Great! Now do 9 * 9"}
        ]

        conversation = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            include_generation_prompt=True
        )

        # Validate conversation structure
        assert len(conversation.token_ids) > 0
        assert conversation.metadata["message_count"] == 5
        assert conversation.metadata["has_system_prompt"] is True
        assert conversation.metadata["has_developer_prompt"] is True

        # Simulate response
        mock_response = create_mock_harmony_response(
            analysis="9 * 9 = 81 (9 squared)",
            final="9 * 9 = 81"
        )
        response_tokens = encode_mock_response(mock_response)

        # Parse and validate
        parsed = parser.parse_response_tokens(response_tokens)
        assert parsed.final == "9 * 9 = 81"
        assert "9 * 9 = 81" in parsed.analysis

    def test_developer_prompt_with_tools_round_trip(self, builder, parser, encoding):
        """Developer prompt with tools: build → parse workflow."""
        # Build developer prompt with tools
        tools = [
            {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
        ]

        developer_prompt = builder.build_developer_prompt(
            instructions="Use the calculate tool when needed.",
            function_tools=tools
        )

        # Verify prompt built correctly
        assert len(developer_prompt.token_ids) > 0
        assert developer_prompt.metadata["has_tools"] is True
        assert developer_prompt.metadata["tool_count"] == 1
        assert "calculate" in developer_prompt.text

        # Build conversation with developer prompt
        messages = [{"role": "user", "content": "What is 100 * 200?"}]
        conversation = builder.build_conversation(
            messages=messages,
            developer_prompt=developer_prompt
        )

        assert len(conversation.token_ids) > 0

    def test_empty_final_channel_round_trip(self, builder, parser, encoding):
        """Test round-trip with empty final channel."""
        # Build simple conversation
        messages = [{"role": "user", "content": "Hello"}]
        conversation = builder.build_conversation(messages=messages)

        # Mock response with empty final
        mock_response = create_mock_harmony_response(
            analysis="User said hello",
            final=""
        )
        response_tokens = encode_mock_response(mock_response)

        # Parse response
        parsed = parser.parse_response_tokens(response_tokens)

        # final should be empty string, not None
        assert parsed.final == ""
        assert parsed.final is not None


# ============================================================================
# B. CONTRACT COMPLIANCE TESTS
# ============================================================================

class TestContractCompliance:
    """Validate contract adherence."""

    def test_builder_contract_compliance(self):
        """Builder implements all contract methods correctly."""
        builder = HarmonyPromptBuilder()

        # Verify builder implements interface
        assert isinstance(builder, HarmonyPromptBuilderInterface)

        # Verify all interface methods exist
        assert hasattr(builder, 'build_system_prompt')
        assert hasattr(builder, 'build_developer_prompt')
        assert hasattr(builder, 'build_conversation')

        # Verify methods are callable
        assert callable(builder.build_system_prompt)
        assert callable(builder.build_developer_prompt)
        assert callable(builder.build_conversation)

    def test_parser_contract_compliance(self):
        """Parser implements all contract methods correctly."""
        parser = HarmonyResponseParser()

        # Verify parser implements interface
        assert isinstance(parser, HarmonyResponseParserInterface)

        # Verify all interface methods exist
        assert hasattr(parser, 'parse_response_tokens')
        assert hasattr(parser, 'parse_response_text')
        assert hasattr(parser, 'validate_harmony_format')
        assert hasattr(parser, 'extract_channel')

        # Verify methods are callable
        assert callable(parser.parse_response_tokens)
        assert callable(parser.parse_response_text)
        assert callable(parser.validate_harmony_format)
        assert callable(parser.extract_channel)

    def test_builder_return_types(self):
        """Builder returns correct types per contract."""
        builder = HarmonyPromptBuilder()

        # Test system prompt return type
        result = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )
        assert isinstance(result, HarmonyPrompt)
        assert isinstance(result.token_ids, list)
        assert isinstance(result.text, str)
        assert isinstance(result.metadata, dict)

        # Test developer prompt return type
        result = builder.build_developer_prompt(instructions="Test")
        assert isinstance(result, HarmonyPrompt)

        # Test conversation return type
        result = builder.build_conversation(messages=[{"role": "user", "content": "Hi"}])
        assert isinstance(result, HarmonyPrompt)

    def test_parser_return_types(self):
        """Parser returns correct types per contract."""
        parser = HarmonyResponseParser()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        # Test parse_response_tokens return type
        mock_response = create_mock_harmony_response(final="Test")
        tokens = encode_mock_response(mock_response)
        result = parser.parse_response_tokens(tokens)

        assert isinstance(result, ParsedHarmonyResponse)
        assert isinstance(result.final, str)
        assert result.final is not None  # NEVER None per contract

    def test_parser_final_never_none(self):
        """Parser NEVER returns None for final field (contract requirement)."""
        parser = HarmonyResponseParser()

        # Test with valid response
        mock_response = create_mock_harmony_response(final="Test")
        tokens = encode_mock_response(mock_response)
        parsed = parser.parse_response_tokens(tokens)
        assert parsed.final is not None

        # Test with empty final
        mock_response = create_mock_harmony_response(final="")
        tokens = encode_mock_response(mock_response)
        parsed = parser.parse_response_tokens(tokens)
        assert parsed.final is not None
        assert parsed.final == ""

        # Test with malformed tokens (should return empty string, not None)
        malformed_tokens = [999999, 888888, 777777]
        parsed = parser.parse_response_tokens(malformed_tokens)
        assert parsed.final is not None
        assert isinstance(parsed.final, str)

    def test_dataclass_contracts(self):
        """Validate HarmonyPrompt and ParsedHarmonyResponse dataclasses."""
        # Test HarmonyPrompt structure
        prompt = HarmonyPrompt(
            token_ids=[1, 2, 3],
            text="test",
            metadata={"key": "value"}
        )
        assert hasattr(prompt, 'token_ids')
        assert hasattr(prompt, 'text')
        assert hasattr(prompt, 'metadata')

        # Test ParsedHarmonyResponse structure
        parsed = ParsedHarmonyResponse(
            final="final text",
            analysis="analysis text",
            commentary="commentary text"
        )
        assert hasattr(parsed, 'final')
        assert hasattr(parsed, 'analysis')
        assert hasattr(parsed, 'commentary')
        assert hasattr(parsed, 'channels')
        assert hasattr(parsed, 'metadata')


# ============================================================================
# C. CROSS-COMPONENT INTEGRATION
# ============================================================================

class TestCrossComponentIntegration:
    """Test integration between builder and parser."""

    @pytest.fixture
    def builder(self):
        return HarmonyPromptBuilder()

    @pytest.fixture
    def parser(self):
        return HarmonyResponseParser()

    @pytest.fixture
    def encoding(self):
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def test_builder_output_parser_input(self, builder, parser, encoding):
        """Builder output format compatible with parser input."""
        # Build conversation
        messages = [{"role": "user", "content": "Test message"}]
        conversation = builder.build_conversation(messages=messages)

        # Validate token_ids format
        assert isinstance(conversation.token_ids, list)
        assert all(isinstance(tid, int) for tid in conversation.token_ids)
        assert all(tid >= 0 for tid in conversation.token_ids)

        # Confirm parser can process these tokens (validate format)
        # Note: conversation tokens are input, not output
        # But we can verify the format is compatible
        assert len(conversation.token_ids) > 0

    def test_complete_workflow(self, builder, parser, encoding):
        """Complete workflow: build → simulate model → parse → display."""
        # Build system prompt
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        # Build developer prompt
        developer_prompt = builder.build_developer_prompt(
            instructions="You are a helpful assistant that provides detailed explanations."
        )

        # Build conversation
        messages = [
            {"role": "user", "content": "Explain photosynthesis"}
        ]

        conversation = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt,
            include_generation_prompt=True
        )

        # Verify conversation built correctly
        assert len(conversation.token_ids) > 0
        assert conversation.metadata["has_system_prompt"] is True
        assert conversation.metadata["has_developer_prompt"] is True

        # Mock model response with all channels
        mock_response = create_mock_harmony_response(
            analysis="Photosynthesis is a complex process. I should explain it clearly.",
            commentary="This is a biology question requiring detailed explanation.",
            final="Photosynthesis is the process by which plants convert light energy into chemical energy."
        )
        response_tokens = encode_mock_response(mock_response)

        # Parse response
        parsed = parser.parse_response_tokens(response_tokens)

        # Validate all channels extracted
        assert parsed.final is not None
        assert parsed.analysis is not None
        assert parsed.commentary is not None
        assert "Photosynthesis" in parsed.final
        assert "complex process" in parsed.analysis
        assert "biology question" in parsed.commentary

        # Simulate display of final channel (production use case)
        display_text = parsed.final
        assert isinstance(display_text, str)
        assert len(display_text) > 0

    def test_error_propagation(self, builder, parser):
        """Errors propagate correctly between components."""
        # Test invalid inputs to builder
        with pytest.raises(ValueError, match="knowledge_cutoff cannot be empty"):
            builder.build_system_prompt(
                reasoning_level=ReasoningLevel.MEDIUM,
                knowledge_cutoff="",
                current_date="2025-10-27"
            )

        with pytest.raises(ValueError, match="instructions cannot be empty"):
            builder.build_developer_prompt(instructions="")

        with pytest.raises(ValueError, match="messages must be a list"):
            builder.build_conversation(messages="not a list")

        # Test invalid inputs to parser
        with pytest.raises(ValueError, match="Token IDs cannot be empty"):
            parser.parse_response_tokens([])

        with pytest.raises(ValueError, match="Response text cannot be empty"):
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            parser.parse_response_text("", encoding)

    def test_metadata_flow(self, builder, parser, encoding):
        """Metadata flows correctly through pipeline."""
        # Build with metadata
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        # Verify builder metadata
        assert system_prompt.metadata["reasoning_level"] == "high"
        assert system_prompt.metadata["include_function_tools"] is True

        # Parse response with metadata
        mock_response = create_mock_harmony_response(final="Test")
        tokens = encode_mock_response(mock_response)
        parsed = parser.parse_response_tokens(tokens)

        # Verify parser metadata
        assert "token_count" in parsed.metadata
        assert "parse_time_ms" in parsed.metadata
        assert parsed.metadata["token_count"] == len(tokens)


# ============================================================================
# D. PERFORMANCE INTEGRATION TESTS
# ============================================================================

class TestPerformanceIntegration:
    """Test performance of combined operations."""

    @pytest.fixture
    def builder(self):
        return HarmonyPromptBuilder()

    @pytest.fixture
    def parser(self):
        return HarmonyResponseParser()

    @pytest.fixture
    def encoding(self):
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def test_end_to_end_latency(self, builder, parser, encoding):
        """Measure total build + parse latency."""
        # Warm up
        for _ in range(3):
            messages = [{"role": "user", "content": "Hello"}]
            conversation = builder.build_conversation(messages=messages)
            mock_response = create_mock_harmony_response(final="Hi")
            tokens = encode_mock_response(mock_response)
            parser.parse_response_tokens(tokens)

        # Measure build + parse time
        iterations = 10
        total_times = []

        for _ in range(iterations):
            start = time.perf_counter()

            # Build conversation
            messages = [{"role": "user", "content": "What is the weather?"}]
            conversation = builder.build_conversation(messages=messages)

            # Parse response
            mock_response = create_mock_harmony_response(
                final="I don't have access to weather data."
            )
            tokens = encode_mock_response(mock_response)
            parsed = parser.parse_response_tokens(tokens)

            elapsed = (time.perf_counter() - start) * 1000  # ms
            total_times.append(elapsed)

        avg_time = sum(total_times) / len(total_times)

        # Total should be reasonable (<200ms for simple conversation)
        # Note: Including encoding overhead, 200ms is reasonable for E2E
        assert avg_time < 200, f"E2E latency {avg_time:.2f}ms exceeds 200ms"

    def test_large_conversation_performance(self, builder, parser, encoding):
        """Test performance with large conversations."""
        # Build conversation with 50+ messages
        messages = []
        for i in range(50):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        # Measure build time
        start = time.perf_counter()
        conversation = builder.build_conversation(messages=messages)
        build_time = (time.perf_counter() - start) * 1000

        # Should build in reasonable time (contract: <10ms per message)
        messages_count = len(messages)
        build_time_per_message = build_time / messages_count
        assert build_time_per_message < BUILDER_PERF["build_conversation"]["max_latency_ms"]

        # Parse large response (5KB+)
        large_content = "A" * 5000
        mock_response = create_mock_harmony_response(final=large_content)
        tokens = encode_mock_response(mock_response)

        # Measure parse time
        start = time.perf_counter()
        parsed = parser.parse_response_tokens(tokens)
        parse_time = (time.perf_counter() - start) * 1000

        # Should parse in reasonable time (contract: <5ms per 1KB)
        size_kb = len(large_content) / 1000
        parse_time_per_kb = parse_time / size_kb
        assert parse_time_per_kb < PARSER_PERF["parse_response_tokens"]["max_latency_ms"]

    def test_concurrent_operations(self, builder, parser, encoding):
        """Test thread safety under concurrent load."""
        def build_and_parse(iteration: int):
            """Build and parse in a thread."""
            # Build conversation
            messages = [{"role": "user", "content": f"Message {iteration}"}]
            conversation = builder.build_conversation(messages=messages)

            # Parse response
            mock_response = create_mock_harmony_response(
                final=f"Response {iteration}"
            )
            tokens = encode_mock_response(mock_response)
            parsed = parser.parse_response_tokens(tokens)

            return (conversation, parsed)

        # Run 20 concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(build_and_parse, i) for i in range(20)]

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Concurrent operation failed: {e}")

        # Verify all operations completed successfully
        assert len(results) == 20

        # Verify no corruption (each result should be unique)
        for i, (conv, parsed) in enumerate(results):
            assert len(conv.token_ids) > 0
            assert parsed.final is not None


# ============================================================================
# E. DATA FLOW VALIDATION
# ============================================================================

class TestDataFlowValidation:
    """Validate data flow through components."""

    @pytest.fixture
    def builder(self):
        return HarmonyPromptBuilder()

    @pytest.fixture
    def parser(self):
        return HarmonyResponseParser()

    @pytest.fixture
    def encoding(self):
        return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def test_reasoning_level_propagation(self, builder, encoding):
        """Reasoning level flows from config to prompt."""
        # Build with ReasoningLevel.HIGH
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.HIGH,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        # Verify metadata contains reasoning level
        assert system_prompt.metadata["reasoning_level"] == "high"

        # Verify token encoding correct
        assert len(system_prompt.token_ids) > 0
        assert all(isinstance(tid, int) for tid in system_prompt.token_ids)

    def test_channel_separation(self, parser, encoding):
        """Channels properly separated in parsing."""
        # Mock response with all three channels
        mock_response = create_mock_harmony_response(
            analysis="This is the analysis channel",
            commentary="This is the commentary channel",
            final="This is the final channel"
        )
        tokens = encode_mock_response(mock_response)

        # Parse response
        parsed = parser.parse_response_tokens(tokens)

        # Verify analysis != commentary != final
        assert parsed.analysis != parsed.commentary
        assert parsed.commentary != parsed.final
        assert parsed.analysis != parsed.final

        # Verify no channel mixing
        assert "analysis channel" in parsed.analysis
        assert "commentary channel" in parsed.commentary
        assert "final channel" in parsed.final

    def test_metadata_preservation(self, builder, parser, encoding):
        """Metadata preserved through pipeline."""
        # Build with metadata
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27",
            include_function_tools=True
        )

        # Verify builder metadata accessible
        assert system_prompt.metadata is not None
        assert "reasoning_level" in system_prompt.metadata
        assert "knowledge_cutoff" in system_prompt.metadata

        # Build conversation with system prompt
        messages = [{"role": "user", "content": "Test"}]
        conversation = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt
        )

        # Verify conversation metadata accessible
        assert conversation.metadata is not None
        assert "has_system_prompt" in conversation.metadata
        assert conversation.metadata["has_system_prompt"] is True

        # Parse response
        mock_response = create_mock_harmony_response(final="Test response")
        tokens = encode_mock_response(mock_response)
        parsed = parser.parse_response_tokens(tokens)

        # Verify parser metadata accessible
        assert parsed.metadata is not None
        assert "token_count" in parsed.metadata
        assert "parse_time_ms" in parsed.metadata

    def test_extract_final_only_optimization(self, parser, encoding):
        """extract_final_only optimization works correctly."""
        # Create response with all channels
        mock_response = create_mock_harmony_response(
            analysis="Long analysis that we want to skip",
            commentary="Long commentary that we want to skip",
            final="Short final answer"
        )
        tokens = encode_mock_response(mock_response)

        # Parse with extract_final_only=False (normal mode)
        start = time.perf_counter()
        parsed_full = parser.parse_response_tokens(tokens, extract_final_only=False)
        time_full = time.perf_counter() - start

        # Parse with extract_final_only=True (optimized mode)
        start = time.perf_counter()
        parsed_opt = parser.parse_response_tokens(tokens, extract_final_only=True)
        time_opt = time.perf_counter() - start

        # Both should have final
        assert parsed_full.final == "Short final answer"
        assert parsed_opt.final == "Short final answer"

        # Full mode should have all channels
        assert parsed_full.analysis is not None
        assert parsed_full.commentary is not None

        # Optimized mode may have analysis (depends on token order)
        # but should still extract final correctly

    def test_token_id_compatibility(self, builder):
        """Token IDs are MLX-compatible (List[int])."""
        # Build various prompts
        system_prompt = builder.build_system_prompt(
            reasoning_level=ReasoningLevel.MEDIUM,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        developer_prompt = builder.build_developer_prompt(
            instructions="Test instructions"
        )

        messages = [{"role": "user", "content": "Test"}]
        conversation = builder.build_conversation(
            messages=messages,
            system_prompt=system_prompt,
            developer_prompt=developer_prompt
        )

        # All should produce MLX-compatible token IDs
        for prompt in [system_prompt, developer_prompt, conversation]:
            assert isinstance(prompt.token_ids, list)
            assert all(isinstance(tid, int) for tid in prompt.token_ids)
            assert all(tid >= 0 for tid in prompt.token_ids)

            # Should be convertible to array (MLX requirement)
            import array
            token_array = array.array('i', prompt.token_ids)
            assert len(token_array) == len(prompt.token_ids)

    def test_empty_conversation_handling(self, builder, parser, encoding):
        """Empty conversations handled gracefully."""
        # Test with empty messages list
        # The builder should handle this gracefully (creates conversation with no messages)
        messages = []
        conversation = builder.build_conversation(messages=messages)

        # Should succeed with empty conversation
        assert isinstance(conversation, HarmonyPrompt)
        assert len(conversation.token_ids) > 0  # Still has prompt structure
        assert conversation.metadata["message_count"] == 0

    def test_channel_extraction_convenience_method(self, parser, encoding):
        """Channel extraction convenience method works correctly."""
        # Parse response
        mock_response = create_mock_harmony_response(
            analysis="Analysis content",
            commentary="Commentary content",
            final="Final content"
        )
        tokens = encode_mock_response(mock_response)
        parsed = parser.parse_response_tokens(tokens)

        # Test extract_channel convenience method
        assert parser.extract_channel(parsed, "final") == "Final content"
        assert parser.extract_channel(parsed, "analysis") == "Analysis content"
        assert parser.extract_channel(parsed, "commentary") == "Commentary content"
        assert parser.extract_channel(parsed, "nonexistent") is None
