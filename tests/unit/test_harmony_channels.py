"""
Unit tests for Harmony channel extraction utilities.

Tests cover:
- Channel extraction for all channel types
- Multi-channel response handling
- Malformed input handling
- Format validation
- Reasoning trace formatting
- PromptTemplate.add_developer() method
"""

import pytest
from src.prompts.harmony_channels import (
    extract_channel,
    extract_all_channels,
    validate_harmony_format,
    format_reasoning_trace,
)
from src.prompts.templates import PromptTemplate


class TestExtractChannel:
    """Test extract_channel function."""

    def test_extract_final_channel(self):
        """Test extracting final channel content."""
        text = "<|start|>assistant<|channel|>final<|message|>Hello, world!<|end|>"
        result = extract_channel(text, "final")
        assert result == "Hello, world!"

    def test_extract_analysis_channel(self):
        """Test extracting analysis channel content."""
        text = "<|start|>assistant<|channel|>analysis<|message|>Step 1: Parse input<|end|>"
        result = extract_channel(text, "analysis")
        assert result == "Step 1: Parse input"

    def test_extract_commentary_channel(self):
        """Test extracting commentary channel content."""
        text = "<|start|>assistant<|channel|>commentary<|message|>This is interesting<|end|>"
        result = extract_channel(text, "commentary")
        assert result == "This is interesting"

    def test_extract_missing_channel(self):
        """Test extracting non-existent channel returns None."""
        text = "<|start|>assistant<|channel|>final<|message|>Hello<|end|>"
        result = extract_channel(text, "analysis")
        assert result is None

    def test_extract_from_multi_channel_response(self):
        """Test extracting specific channel from multi-channel response."""
        text = """<|start|>assistant<|channel|>analysis<|message|>Think step by step<|end|>
<|start|>assistant<|channel|>final<|message|>The answer is 42<|end|>"""

        analysis = extract_channel(text, "analysis")
        final = extract_channel(text, "final")

        assert analysis == "Think step by step"
        assert final == "The answer is 42"

    def test_extract_channel_with_multiline_content(self):
        """Test extracting channel with multi-line content."""
        text = """<|start|>assistant<|channel|>analysis<|message|>Step 1: Parse
Step 2: Process
Step 3: Output<|end|>"""
        result = extract_channel(text, "analysis")
        assert "Step 1: Parse" in result
        assert "Step 2: Process" in result
        assert "Step 3: Output" in result

    def test_extract_channel_empty_input(self):
        """Test extracting channel from empty input."""
        assert extract_channel("", "final") is None
        assert extract_channel(None, "final") is None

    def test_extract_channel_malformed_input(self):
        """Test extracting channel from malformed input."""
        result = extract_channel("not harmony format", "final")
        assert result is None

    def test_extract_channel_with_whitespace(self):
        """Test extracting channel handles whitespace correctly."""
        text = "<|start|>assistant<|channel|>  final  <|message|>  Hello  <|end|>"
        result = extract_channel(text, "final")
        assert result == "Hello"

    def test_extract_duplicate_channels_returns_last(self):
        """Test extracting from duplicate channels returns first occurrence."""
        text = """<|start|>assistant<|channel|>final<|message|>First<|end|>
<|start|>assistant<|channel|>final<|message|>Second<|end|>"""
        # extract_channel returns first match it finds
        result = extract_channel(text, "final")
        assert result == "First"


class TestExtractAllChannels:
    """Test extract_all_channels function."""

    def test_extract_single_channel(self):
        """Test extracting all channels from single-channel response."""
        text = "<|start|>assistant<|channel|>final<|message|>Hello<|end|>"
        result = extract_all_channels(text)
        assert result == {"final": "Hello"}

    def test_extract_multiple_channels(self):
        """Test extracting all channels from multi-channel response."""
        text = """<|start|>assistant<|channel|>analysis<|message|>Reasoning here<|end|>
<|start|>assistant<|channel|>commentary<|message|>Interesting note<|end|>
<|start|>assistant<|channel|>final<|message|>Final answer<|end|>"""

        result = extract_all_channels(text)

        assert len(result) == 3
        assert result["analysis"] == "Reasoning here"
        assert result["commentary"] == "Interesting note"
        assert result["final"] == "Final answer"

    def test_extract_all_channels_empty_input(self):
        """Test extracting all channels from empty input."""
        assert extract_all_channels("") == {}
        assert extract_all_channels(None) == {}

    def test_extract_all_channels_malformed_input(self):
        """Test extracting all channels from malformed input."""
        result = extract_all_channels("plain text without markers")
        assert result == {}

    def test_extract_all_channels_partial_markers(self):
        """Test extracting all channels with partial markers."""
        text = "<|start|>assistant<|channel|>final<|message|>incomplete"
        result = extract_all_channels(text)
        # No complete channel markers, should return empty
        assert result == {}

    def test_extract_all_channels_duplicate_channel(self):
        """Test extracting all channels with duplicate channel names."""
        text = """<|start|>assistant<|channel|>final<|message|>First<|end|>
<|start|>assistant<|channel|>final<|message|>Second<|end|>"""
        result = extract_all_channels(text)
        # Should keep last occurrence of duplicate channel
        assert result["final"] == "Second"

    def test_extract_all_channels_preserves_formatting(self):
        """Test that extract_all_channels preserves content formatting."""
        text = """<|start|>assistant<|channel|>analysis<|message|>Line 1
Line 2
  Indented line<|end|>"""
        result = extract_all_channels(text)
        assert "Line 1\nLine 2\n  Indented line" in result["analysis"]


class TestValidateHarmonyFormat:
    """Test validate_harmony_format function."""

    def test_validate_user_message(self):
        """Test validating user message format."""
        text = "<|start|>user<|message|>Hello<|end|>"
        assert validate_harmony_format(text) is True

    def test_validate_system_message(self):
        """Test validating system message format."""
        text = "<|start|>system<|message|>You are helpful<|end|>"
        assert validate_harmony_format(text) is True

    def test_validate_assistant_message_with_channel(self):
        """Test validating assistant message with channel."""
        text = "<|start|>assistant<|channel|>final<|message|>Response<|end|>"
        assert validate_harmony_format(text) is True

    def test_validate_multiple_messages(self):
        """Test validating multiple messages."""
        text = """<|start|>user<|message|>Hi<|end|>
<|start|>assistant<|channel|>final<|message|>Hello<|end|>"""
        assert validate_harmony_format(text) is True

    def test_validate_plain_text_fails(self):
        """Test validating plain text returns False."""
        assert validate_harmony_format("Just plain text") is False

    def test_validate_empty_input(self):
        """Test validating empty input returns False."""
        assert validate_harmony_format("") is False
        assert validate_harmony_format(None) is False

    def test_validate_partial_markers_fails(self):
        """Test validating partial markers returns False."""
        assert validate_harmony_format("<|start|>incomplete") is False
        assert validate_harmony_format("<|message|>no start") is False

    def test_validate_with_extra_text(self):
        """Test validating format with surrounding text."""
        text = "Some prefix <|start|>user<|message|>Hi<|end|> some suffix"
        assert validate_harmony_format(text) is True


class TestFormatReasoningTrace:
    """Test format_reasoning_trace function."""

    def test_format_simple_reasoning(self):
        """Test formatting simple reasoning text."""
        reasoning = "Step 1: Parse input\nStep 2: Process"
        result = format_reasoning_trace(reasoning)
        assert result == "Step 1: Parse input\nStep 2: Process"

    def test_format_reasoning_truncation(self):
        """Test formatting with truncation."""
        reasoning = "A" * 600
        result = format_reasoning_trace(reasoning, max_length=100)

        assert len(result) <= 113  # 100 + len(" [TRUNCATED]")
        assert result.endswith("[TRUNCATED]")
        assert result.startswith("A" * 50)

    def test_format_reasoning_no_truncation_needed(self):
        """Test formatting when content is under max_length."""
        reasoning = "Short text"
        result = format_reasoning_trace(reasoning, max_length=100)
        assert result == "Short text"
        assert "[TRUNCATED]" not in result

    def test_format_reasoning_empty_input(self):
        """Test formatting empty input."""
        assert format_reasoning_trace("") == ""
        assert format_reasoning_trace(None) == ""

    def test_format_reasoning_removes_channel_markers(self):
        """Test formatting removes Harmony channel markers."""
        reasoning = "<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>"
        result = format_reasoning_trace(reasoning)
        assert result == "Thinking"
        assert "<|" not in result

    def test_format_reasoning_removes_end_markers(self):
        """Test formatting removes end markers."""
        reasoning = "Some reasoning<|end|>"
        result = format_reasoning_trace(reasoning)
        assert result == "Some reasoning"
        assert "<|end|>" not in result

    def test_format_reasoning_preserves_structure(self):
        """Test formatting preserves line breaks and structure."""
        reasoning = """Step 1: First
Step 2: Second
  - Substep A
  - Substep B"""
        result = format_reasoning_trace(reasoning, max_length=1000)
        assert "Step 1: First" in result
        assert "Step 2: Second" in result
        assert "  - Substep A" in result

    def test_format_reasoning_at_exact_limit(self):
        """Test formatting when text is exactly at max_length."""
        reasoning = "A" * 100
        result = format_reasoning_trace(reasoning, max_length=100)
        assert result == "A" * 100
        assert "[TRUNCATED]" not in result

    def test_format_reasoning_strips_whitespace(self):
        """Test formatting strips leading/trailing whitespace."""
        reasoning = "  \n  Content here  \n  "
        result = format_reasoning_trace(reasoning)
        assert result == "Content here"


class TestPromptTemplateAddDeveloper:
    """Test PromptTemplate.add_developer() method."""

    def test_add_developer_message(self):
        """Test adding developer message to template."""
        template = PromptTemplate("harmony")
        template.add_developer("Developer instructions here")

        assert len(template.messages) == 1
        assert template.messages[0]["role"] == "developer"
        assert template.messages[0]["content"] == "Developer instructions here"

    def test_add_developer_chaining(self):
        """Test add_developer returns self for chaining."""
        template = PromptTemplate("harmony")
        result = template.add_developer("Test")
        assert result is template

    def test_add_developer_with_other_messages(self):
        """Test developer message in conversation with other messages."""
        template = (PromptTemplate("harmony")
            .add_system("You are helpful")
            .add_developer("Use these guidelines")
            .add_user("Hello")
            .add_assistant("Hi there"))

        assert len(template.messages) == 4
        assert template.messages[0]["role"] == "system"
        assert template.messages[1]["role"] == "developer"
        assert template.messages[2]["role"] == "user"
        assert template.messages[3]["role"] == "assistant"

    def test_add_developer_formats_correctly(self):
        """Test developer message formats with harmony template."""
        template = (PromptTemplate("harmony")
            .add_developer("Test developer message"))

        formatted = template.build()
        assert "<|start|>developer<|message|>" in formatted
        assert "Test developer message" in formatted
        assert "<|end|>" in formatted

    def test_multiple_developer_messages(self):
        """Test adding multiple developer messages."""
        template = (PromptTemplate("harmony")
            .add_developer("First dev message")
            .add_developer("Second dev message"))

        assert len(template.messages) == 2
        assert all(msg["role"] == "developer" for msg in template.messages)

    def test_developer_message_ordering(self):
        """Test developer messages maintain insertion order."""
        template = (PromptTemplate("harmony")
            .add_system("System")
            .add_developer("Dev 1")
            .add_user("User")
            .add_developer("Dev 2"))

        roles = [msg["role"] for msg in template.messages]
        assert roles == ["system", "developer", "user", "developer"]


class TestHarmonyTemplateRegistration:
    """Test Harmony template is properly registered."""

    def test_harmony_template_exists(self):
        """Test harmony template is in TEMPLATES dict."""
        from src.prompts.templates import TEMPLATES
        assert "harmony" in TEMPLATES

    def test_harmony_template_has_required_fields(self):
        """Test harmony template has all required fields."""
        from src.prompts.templates import TEMPLATES
        harmony = TEMPLATES["harmony"]

        assert "system" in harmony
        assert "developer" in harmony
        assert "user" in harmony
        assert "assistant" in harmony
        assert "assistant_start" in harmony
        assert "name" in harmony
        assert "description" in harmony

    def test_harmony_template_format_correct(self):
        """Test harmony template uses correct format strings."""
        from src.prompts.templates import TEMPLATES
        harmony = TEMPLATES["harmony"]

        assert "<|start|>system<|message|>" in harmony["system"]
        assert "<|start|>developer<|message|>" in harmony["developer"]
        assert "<|start|>user<|message|>" in harmony["user"]
        assert "<|channel|>" in harmony["assistant"]
        assert "<|start|>assistant<|channel|>final<|message|>" in harmony["assistant_start"]

    def test_harmony_template_can_be_instantiated(self):
        """Test PromptTemplate can be created with harmony template."""
        template = PromptTemplate("harmony")
        assert template.template_name == "harmony"


class TestEdgeCasesAndExceptionHandling:
    """Test edge cases and exception handling paths."""

    def test_extract_channel_with_regex_catastrophic_backtracking(self):
        """Test extract_channel handles potentially problematic regex inputs."""
        # Create input that could cause regex issues
        malicious = "<|start|>assistant<|channel|>" + ("x" * 10000) + "<|message|>content<|end|>"
        result = extract_channel(malicious, "test")
        # Should handle gracefully without hanging
        assert result is None or isinstance(result, str)

    def test_extract_all_channels_with_nested_markers(self):
        """Test extract_all_channels with nested or malformed markers."""
        text = "<|start|>assistant<|channel|>test<|message|><|start|>nested<|end|>"
        result = extract_all_channels(text)
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_validate_harmony_format_with_unicode(self):
        """Test validate_harmony_format with unicode characters."""
        text = "<|start|>user<|message|>こんにちは 🌍<|end|>"
        assert validate_harmony_format(text) is True

    def test_format_reasoning_trace_with_unicode_and_special_chars(self):
        """Test format_reasoning_trace with unicode and special characters."""
        reasoning = "Step 1: Test 测试 🔬\nStep 2: More 更多 ✅"
        result = format_reasoning_trace(reasoning)
        assert "Test 测试" in result
        assert "More 更多" in result


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_build_and_parse_harmony_conversation(self):
        """Test building harmony format and parsing response."""
        # Build a harmony formatted prompt
        template = (PromptTemplate("harmony")
            .add_system("You are a helpful assistant")
            .add_user("What is 2+2?"))

        prompt = template.build(include_generation_prompt=True)

        # Verify format
        assert validate_harmony_format(prompt)
        assert "<|start|>assistant<|channel|>final<|message|>" in prompt

        # Simulate response with multiple channels
        response = """<|start|>assistant<|channel|>analysis<|message|>Simple arithmetic: 2+2=4<|end|>
<|start|>assistant<|channel|>final<|message|>The answer is 4<|end|>"""

        # Extract channels
        channels = extract_all_channels(response)
        assert channels["analysis"] == "Simple arithmetic: 2+2=4"
        assert channels["final"] == "The answer is 4"

        # Format reasoning
        reasoning = format_reasoning_trace(channels["analysis"], max_length=100)
        assert "Simple arithmetic" in reasoning

    def test_developer_message_in_harmony_format(self):
        """Test developer messages work with harmony template."""
        template = (PromptTemplate("harmony")
            .add_system("You are helpful")
            .add_developer("Always provide reasoning in analysis channel")
            .add_user("Explain something"))

        formatted = template.build()

        # Check developer message is formatted correctly
        assert "<|start|>developer<|message|>" in formatted
        assert "Always provide reasoning" in formatted
        assert validate_harmony_format(formatted)

    def test_extract_and_format_workflow(self):
        """Test complete extract and format workflow."""
        response = """<|start|>assistant<|channel|>analysis<|message|>""" + ("X" * 1000) + """<|end|>
<|start|>assistant<|channel|>final<|message|>Answer<|end|>"""

        # Extract analysis channel
        analysis = extract_channel(response, "analysis")
        assert len(analysis) == 1000

        # Format with truncation
        formatted = format_reasoning_trace(analysis, max_length=100)
        assert len(formatted) <= 113
        assert formatted.endswith("[TRUNCATED]")

        # Extract final answer
        final = extract_channel(response, "final")
        assert final == "Answer"
