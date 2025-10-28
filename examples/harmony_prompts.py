#!/usr/bin/env python3
"""
Harmony Format Examples - openai-harmony Implementation

This file demonstrates how to use the new Harmony format implementation
with the official openai-harmony package.

Examples:
1. Basic Usage - Minimal setup and usage
2. Reasoning Levels - LOW/MEDIUM/HIGH comparison
3. Multi-Turn Conversations - Conversation history
4. Presets Integration - Using preset helpers
5. Streaming with Channels - Real-time parsing
6. Complete Workflow - End-to-end example

Requirements:
- openai-harmony >= 0.0.4
- MLX-based inference engine

Author: tkr-gpt-oss-20b
Updated: 2025-10-27 (Harmony Replacement - Wave 5)
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prompts.harmony_native import (
    HarmonyPromptBuilder,
    HarmonyResponseParser,
    ReasoningLevel,
    HarmonyPrompt,
    ParsedHarmonyResponse
)
from src.prompts.presets import get_preset, get_developer_content

# For demonstration purposes, we'll use mock tokenization and responses
# In production, use the actual MLX tokenizer and inference engine


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)


def example_1_basic_usage():
    """
    Example 1: Basic Usage - Minimal Setup and Usage

    Demonstrates:
    - Creating a HarmonyPromptBuilder
    - Building a system prompt with default reasoning level
    - Building a simple conversation
    - Basic usage without complexity

    This is the simplest way to use Harmony format.
    """
    print_section("Example 1: Basic Usage")

    # Step 1: Create the builder
    builder = HarmonyPromptBuilder()
    print("✓ Created HarmonyPromptBuilder")

    # Step 2: Build system prompt with default (MEDIUM) reasoning
    system_prompt = builder.build_system_prompt(
        reasoning_level=ReasoningLevel.MEDIUM,
        knowledge_cutoff="2024-06",
        current_date="2025-10-27"
    )

    print(f"\n✓ Built system prompt:")
    print(f"  - Tokens: {len(system_prompt.token_ids)}")
    print(f"  - Reasoning level: {system_prompt.metadata['reasoning_level']}")
    print(f"  - Text preview: {system_prompt.text[:100]}...")

    # Step 3: Build a simple conversation
    messages = [
        {"role": "user", "content": "What is Python?"}
    ]

    conversation = builder.build_conversation(
        messages=messages,
        system_prompt=system_prompt,
        include_generation_prompt=True
    )

    print(f"\n✓ Built conversation:")
    print(f"  - Total tokens: {len(conversation.token_ids)}")
    print(f"  - Messages: {conversation.metadata['message_count']}")
    print(f"  - Has generation prompt: {conversation.metadata['include_generation_prompt']}")

    # Step 4: Simulate response parsing
    print("\n✓ In production, you would:")
    print("  1. Pass conversation.token_ids to MLX model")
    print("  2. Collect generated token IDs")
    print("  3. Parse response with HarmonyResponseParser")

    # Mock response parsing example
    print_subsection("Response Parsing (Conceptual)")

    print("\n💡 In production, you would:")
    print("  1. Get token IDs from model generation")
    print("  2. Create HarmonyResponseParser instance")
    print("  3. Call parser.parse_response_tokens(token_ids)")
    print("  4. Access parsed.final for user-facing content")
    print("  5. Access parsed.analysis for debugging")

    print("\n📝 Example Code:")
    print("""
    # Production parsing example:
    parser = HarmonyResponseParser()
    parsed = parser.parse_response_tokens(
        token_ids=generated_token_ids,
        extract_final_only=False  # Get all channels
    )

    # Display to user
    print(parsed.final)

    # Log for debugging
    if parsed.analysis:
        logger.debug(f"Reasoning: {parsed.analysis}")
    """)

    print("\n✓ Expected response structure:")
    print("  - Analysis channel: Model's reasoning process")
    print("    Example: 'Python is a programming language. I should provide...'")
    print("  - Final channel: User-facing response")
    print("    Example: 'Python is a high-level, interpreted programming language...'")
    print("  - Metadata: Parse time, token count, etc.")

    print("\n✅ Basic usage complete!")


def example_2_reasoning_levels():
    """
    Example 2: Reasoning Levels - LOW/MEDIUM/HIGH Comparison

    Demonstrates:
    - Different reasoning levels (LOW, MEDIUM, HIGH)
    - How reasoning level affects system prompt
    - Impact on response verbosity
    - When to use each level
    """
    print_section("Example 2: Reasoning Levels")

    builder = HarmonyPromptBuilder()

    print("Comparing reasoning levels:\n")

    levels = [
        (ReasoningLevel.LOW, "Fast, concise responses for simple tasks"),
        (ReasoningLevel.MEDIUM, "Balanced approach for most use cases"),
        (ReasoningLevel.HIGH, "Thorough reasoning for complex problems")
    ]

    for reasoning_level, description in levels:
        print_subsection(f"{reasoning_level.value.upper()} Reasoning")

        system_prompt = builder.build_system_prompt(
            reasoning_level=reasoning_level,
            knowledge_cutoff="2024-06",
            current_date="2025-10-27"
        )

        print(f"Description: {description}")
        print(f"System prompt tokens: {len(system_prompt.token_ids)}")
        print(f"Metadata: {system_prompt.metadata}")

        # Show how reasoning level affects the prompt
        if reasoning_level == ReasoningLevel.LOW:
            print("\n💡 Use LOW for:")
            print("   - Simple factual questions")
            print("   - Quick lookups")
            print("   - When speed is critical")
            print("   - Example: 'What is 2+2?'")
        elif reasoning_level == ReasoningLevel.MEDIUM:
            print("\n💡 Use MEDIUM for:")
            print("   - General conversation")
            print("   - Balanced responses")
            print("   - Most use cases")
            print("   - Example: 'Explain how photosynthesis works'")
        else:  # HIGH
            print("\n💡 Use HIGH for:")
            print("   - Complex analysis")
            print("   - Multi-step problems")
            print("   - When quality > speed")
            print("   - Example: 'Design a scalable microservices architecture'")

    print("\n✅ Reasoning levels comparison complete!")


def example_3_multi_turn_conversation():
    """
    Example 3: Multi-Turn Conversations - Conversation History

    Demonstrates:
    - Building conversations with history
    - Preserving context across turns
    - Handling multi-turn dialogue
    - Message structure for conversation
    """
    print_section("Example 3: Multi-Turn Conversations")

    builder = HarmonyPromptBuilder()

    # Build system prompt
    system_prompt = builder.build_system_prompt(
        reasoning_level=ReasoningLevel.MEDIUM,
        knowledge_cutoff="2024-06",
        current_date="2025-10-27"
    )

    print("Building a multi-turn conversation:\n")

    # Turn 1: Initial question
    print_subsection("Turn 1: Initial Question")

    messages_turn1 = [
        {"role": "user", "content": "What is machine learning?"}
    ]

    conversation_turn1 = builder.build_conversation(
        messages=messages_turn1,
        system_prompt=system_prompt,
        include_generation_prompt=True
    )

    print(f"Messages: {len(messages_turn1)}")
    print(f"Total tokens: {len(conversation_turn1.token_ids)}")
    print("User: What is machine learning?")
    print("Assistant: <generates response>")

    # Turn 2: Follow-up question
    print_subsection("Turn 2: Follow-up Question")

    messages_turn2 = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you give me an example?"}
    ]

    conversation_turn2 = builder.build_conversation(
        messages=messages_turn2,
        system_prompt=system_prompt,
        include_generation_prompt=True
    )

    print(f"Messages: {len(messages_turn2)}")
    print(f"Total tokens: {len(conversation_turn2.token_ids)}")
    print("User: Can you give me an example?")
    print("Assistant: <generates response>")

    # Turn 3: Clarification
    print_subsection("Turn 3: Clarification")

    messages_turn3 = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you give me an example?"},
        {"role": "assistant", "content": "Sure! For example, email spam detection..."},
        {"role": "user", "content": "How does the spam detector learn?"}
    ]

    conversation_turn3 = builder.build_conversation(
        messages=messages_turn3,
        system_prompt=system_prompt,
        include_generation_prompt=True
    )

    print(f"Messages: {len(messages_turn3)}")
    print(f"Total tokens: {len(conversation_turn3.token_ids)}")
    print("User: How does the spam detector learn?")
    print("Assistant: <generates response>")

    print("\n💡 Key Points:")
    print("   - Each turn includes full conversation history")
    print("   - Context is preserved across turns")
    print("   - Token count grows with conversation length")
    print("   - Consider context window limits for long conversations")

    print("\n✅ Multi-turn conversation complete!")


def example_4_presets_integration():
    """
    Example 4: Presets Integration - Using Preset Helpers

    Demonstrates:
    - Using get_developer_content() with presets
    - Different preset personalities
    - Building developer prompts from presets
    - Combining system and developer prompts
    """
    print_section("Example 4: Presets Integration")

    builder = HarmonyPromptBuilder()

    # Build system prompt
    system_prompt = builder.build_system_prompt(
        reasoning_level=ReasoningLevel.MEDIUM,
        knowledge_cutoff="2024-06",
        current_date="2025-10-27"
    )

    print("Demonstrating different presets:\n")

    # Example 1: Coding Assistant
    print_subsection("Preset: Coding Assistant")

    # Get preset text
    coding_preset = get_preset("coding")
    print(f"Preset text: {coding_preset[:80]}...")

    # Build developer prompt
    developer_prompt_coding = builder.build_developer_prompt(
        instructions=coding_preset
    )

    print(f"\n✓ Built developer prompt:")
    print(f"  - Tokens: {len(developer_prompt_coding.token_ids)}")
    print(f"  - Has instructions: {developer_prompt_coding.metadata['has_instructions']}")

    # Build complete conversation
    messages = [{"role": "user", "content": "How do I read a file in Python?"}]

    conversation = builder.build_conversation(
        messages=messages,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt_coding,
        include_generation_prompt=True
    )

    print(f"\n✓ Complete conversation:")
    print(f"  - Total tokens: {len(conversation.token_ids)}")
    print(f"  - Has system prompt: {conversation.metadata['has_system_prompt']}")
    print(f"  - Has developer prompt: {conversation.metadata['has_developer_prompt']}")

    # Example 2: Creative Assistant
    print_subsection("Preset: Creative Assistant")

    creative_preset = get_preset("creative")
    developer_prompt_creative = builder.build_developer_prompt(
        instructions=creative_preset
    )

    print(f"Preset: {creative_preset[:60]}...")
    print(f"Developer prompt tokens: {len(developer_prompt_creative.token_ids)}")

    # Example 3: Analytical Assistant
    print_subsection("Preset: Analytical Assistant")

    analytical_preset = get_preset("analytical")
    developer_prompt_analytical = builder.build_developer_prompt(
        instructions=analytical_preset
    )

    print(f"Preset: {analytical_preset[:60]}...")
    print(f"Developer prompt tokens: {len(developer_prompt_analytical.token_ids)}")

    print("\n💡 Available Presets:")
    available_presets = [
        "default", "concise", "detailed", "coding", "creative",
        "analytical", "teacher", "professional", "casual", "researcher",
        "debug", "minimalist", "socratic", "roleplay"
    ]
    print(f"   {', '.join(available_presets)}")

    print("\n✅ Presets integration complete!")


def example_5_streaming_with_channels():
    """
    Example 5: Streaming with Channels - Real-time Parsing

    Demonstrates:
    - Streaming generation with channel detection
    - Real-time channel metadata
    - Filtering by channel
    - Displaying different channels
    """
    print_section("Example 5: Streaming with Channels")

    print("Simulating streaming generation with channel detection:\n")

    # Simulate streaming tokens from model
    # In production, these would come from MLX model in real-time
    mock_streaming_response = """<|start|>assistant<|channel|>analysis<|message|>
This is a complex question about quantum computing. Let me break it down:
1. Define quantum computing
2. Explain key concepts (qubits, superposition, entanglement)
3. Mention applications
4. Note current limitations
<|end|>
<|start|>assistant<|channel|>commentary<|message|>
Drawing from physics and computer science knowledge bases.
Using simplified explanations for general audience.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform calculations. Unlike classical bits (0 or 1), quantum bits (qubits) can exist in multiple states simultaneously. This enables quantum computers to solve certain problems exponentially faster than classical computers, particularly in cryptography, optimization, and molecular simulation. However, quantum computers are still experimental and face challenges like high error rates and maintaining quantum coherence.
<|end|>"""

    # Simulate streaming by processing text in chunks
    print_subsection("Streaming Simulation")

    print("\n[Starting stream...]")

    # Simulate channel detection during streaming
    current_channel = None
    buffer = ""

    # Simple channel detection (in production, use proper token-based detection)
    lines = mock_streaming_response.split('\n')

    for line in lines:
        if '<|channel|>analysis<|message|>' in line:
            if current_channel:
                print(f"\n[Channel: {current_channel}] {buffer[:50]}...")
            current_channel = "analysis"
            buffer = ""
            print("\n🔍 [Analysis channel detected]")
        elif '<|channel|>commentary<|message|>' in line:
            if current_channel:
                print(f"\n[Channel: {current_channel}] {buffer[:50]}...")
            current_channel = "commentary"
            buffer = ""
            print("\n💬 [Commentary channel detected]")
        elif '<|channel|>final<|message|>' in line:
            if current_channel:
                print(f"\n[Channel: {current_channel}] {buffer[:50]}...")
            current_channel = "final"
            buffer = ""
            print("\n📄 [Final channel detected - user-facing response]")
        elif '<|end|>' in line:
            if current_channel and buffer.strip():
                print(f"[Channel: {current_channel}] Complete")
            current_channel = None
        elif line.strip() and current_channel:
            buffer += line + " "

    print("\n[Stream complete]")

    # Now demonstrate response parsing
    print_subsection("Response Parsing")

    print("\n💡 In production, the parser would extract:")

    # Expected parsed channels
    expected_final = "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform calculations..."
    expected_analysis = "This is a complex question about quantum computing. Let me break it down: 1. Define quantum computing..."
    expected_commentary = "Drawing from physics and computer science knowledge bases. Using simplified explanations for general audience."

    print(f"  ✓ Analysis channel: {expected_analysis[:60]}...")
    print(f"  ✓ Commentary channel: {expected_commentary[:60]}...")
    print(f"  ✓ Final channel: {expected_final[:60]}...")

    # Show each channel separately
    print_subsection("Channel Display Options")

    print("\n1️⃣  Show FINAL only (for end users):")
    print(f"   {expected_final[:100]}...")

    print("\n2️⃣  Show FINAL + ANALYSIS (for developers debugging):")
    print(f"   [Final]: {expected_final[:80]}...")
    print(f"   [Analysis]: {expected_analysis[:80]}...")

    print("\n3️⃣  Show ALL channels (complete transparency):")
    print(f"   [analysis]: {expected_analysis[:60]}...")
    print(f"   [commentary]: {expected_commentary[:60]}...")
    print(f"   [final]: {expected_final[:60]}...")

    print("\n💡 Streaming Benefits:")
    print("   - Real-time channel detection")
    print("   - Progressive display of final response")
    print("   - Live reasoning traces for debugging")
    print("   - Better user experience for long generations")

    print("\n✅ Streaming with channels complete!")


def example_6_complete_workflow():
    """
    Example 6: Complete Workflow - End-to-End Example

    Demonstrates:
    - Complete workflow from configuration to display
    - All features integrated together
    - Error handling and best practices
    - Production-ready example
    """
    print_section("Example 6: Complete Workflow")

    print("End-to-end example with all features:\n")

    # Step 1: Configuration
    print_subsection("Step 1: Configuration")

    # Define configuration
    config = {
        "reasoning_level": ReasoningLevel.HIGH,
        "knowledge_cutoff": "2024-06",
        "current_date": "2025-10-27",
        "preset": "analytical",
        "include_function_tools": False
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    # Step 2: Build Prompts
    print_subsection("Step 2: Build Prompts")

    builder = HarmonyPromptBuilder()

    # Build system prompt
    system_prompt = builder.build_system_prompt(
        reasoning_level=config["reasoning_level"],
        knowledge_cutoff=config["knowledge_cutoff"],
        current_date=config["current_date"],
        include_function_tools=config["include_function_tools"]
    )

    print(f"✓ System prompt: {len(system_prompt.token_ids)} tokens")

    # Build developer prompt from preset
    preset_text = get_preset(config["preset"])
    developer_prompt = builder.build_developer_prompt(
        instructions=preset_text
    )

    print(f"✓ Developer prompt: {len(developer_prompt.token_ids)} tokens")

    # Step 3: Build Conversation
    print_subsection("Step 3: Build Conversation")

    # Multi-turn conversation
    messages = [
        {"role": "user", "content": "Analyze the pros and cons of remote work."},
        {"role": "assistant", "content": "Remote work has several advantages and disadvantages..."},
        {"role": "user", "content": "What about the impact on team collaboration?"}
    ]

    conversation = builder.build_conversation(
        messages=messages,
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        include_generation_prompt=True
    )

    print(f"✓ Conversation: {len(conversation.token_ids)} tokens")
    print(f"  - Messages: {conversation.metadata['message_count']}")
    print(f"  - Has system: {conversation.metadata['has_system_prompt']}")
    print(f"  - Has developer: {conversation.metadata['has_developer_prompt']}")

    # Step 4: Generate (Mock)
    print_subsection("Step 4: Generate Response")

    print("\n[In production: Pass conversation.token_ids to MLX model]")
    print("[Collecting generated tokens...]")

    # Mock response
    mock_response = """<|start|>assistant<|channel|>analysis<|message|>
Analyzing team collaboration in remote work context:
1. Synchronous vs asynchronous communication
2. Tools and technology requirements
3. Cultural and timezone challenges
4. Trust and accountability factors
5. Creative collaboration considerations

Need to present balanced view with specific examples.
<|end|>
<|start|>assistant<|channel|>commentary<|message|>
Using analytical framework for structured response.
Drawing on organizational behavior research.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Remote work significantly impacts team collaboration in several ways:

**Challenges:**
- **Reduced spontaneous interaction:** No hallway conversations or impromptu brainstorming
- **Communication barriers:** Harder to read body language and social cues
- **Timezone coordination:** Difficult for globally distributed teams
- **Technology dependency:** Requires reliable tools and internet connectivity
- **Team bonding:** Less natural relationship building

**Opportunities:**
- **Asynchronous collaboration:** Team members can contribute when most productive
- **Documented communication:** Written records improve clarity and accountability
- **Diverse perspectives:** Access to global talent pool
- **Focused work:** Fewer interruptions for deep work
- **Inclusive participation:** Digital tools can amplify quieter voices

**Best Practices:**
- Use a mix of synchronous (video calls) and asynchronous (documents, chat) methods
- Establish clear communication norms and response expectations
- Invest in collaborative tools (Slack, Miro, Notion)
- Schedule regular team building activities
- Create dedicated channels for social interaction
<|end|>"""

    print("[Generation complete]")

    # Step 5: Parse Response
    print_subsection("Step 5: Parse Response")

    print("\n💡 In production:")
    print("   parser = HarmonyResponseParser()")
    print("   parsed = parser.parse_response_tokens(generated_token_ids)")

    # Expected parsed content
    expected_final = """Remote work significantly impacts team collaboration in several ways:

**Challenges:**
- **Reduced spontaneous interaction:** No hallway conversations or impromptu brainstorming
- **Communication barriers:** Harder to read body language and social cues
- **Timezone coordination:** Difficult for globally distributed teams
- **Technology dependency:** Requires reliable tools and internet connectivity
- **Team bonding:** Less natural relationship building

**Opportunities:**
- **Asynchronous collaboration:** Team members can contribute when most productive
- **Documented communication:** Written records improve clarity and accountability
- **Diverse perspectives:** Access to global talent pool
- **Focused work:** Fewer interruptions for deep work
- **Inclusive participation:** Digital tools can amplify quieter voices

**Best Practices:**
- Use a mix of synchronous (video calls) and asynchronous (documents, chat) methods
- Establish clear communication norms and response expectations
- Invest in collaborative tools (Slack, Miro, Notion)
- Schedule regular team building activities
- Create dedicated channels for social interaction"""

    expected_analysis = """Analyzing team collaboration in remote work context:
1. Synchronous vs asynchronous communication
2. Tools and technology requirements
3. Cultural and timezone challenges
4. Trust and accountability factors
5. Creative collaboration considerations

Need to present balanced view with specific examples."""

    expected_commentary = """Using analytical framework for structured response.
Drawing on organizational behavior research."""

    print(f"\n✓ Parsing complete:")
    print(f"  - Final: {len(expected_final)} chars")
    print(f"  - Analysis: {len(expected_analysis)} chars")
    print(f"  - Commentary: {len(expected_commentary)} chars")

    # Step 6: Display Results
    print_subsection("Step 6: Display Results")

    print("\n📄 USER-FACING RESPONSE:")
    print("-" * 70)
    print(expected_final)
    print("-" * 70)

    print("\n🔍 DEVELOPER DEBUG INFO:")
    print(f"Analysis channel: {expected_analysis[:200]}...")
    print(f"Commentary channel: {expected_commentary}")

    print("\n📊 METADATA:")
    print(f"  - final_length: {len(expected_final)} chars")
    print(f"  - analysis_length: {len(expected_analysis)} chars")
    print(f"  - commentary_length: {len(expected_commentary)} chars")
    print(f"  - channels_detected: 3 (analysis, commentary, final)")

    # Step 7: Best Practices Summary
    print_subsection("Step 7: Best Practices")

    print("\n✅ Production Checklist:")
    print("   [✓] Use appropriate reasoning level for task")
    print("   [✓] Include system prompt with context")
    print("   [✓] Choose preset matching use case")
    print("   [✓] Build conversation with proper message structure")
    print("   [✓] Handle errors gracefully")
    print("   [✓] Parse response with HarmonyResponseParser")
    print("   [✓] Show FINAL channel to users")
    print("   [✓] Log ANALYSIS channel for debugging")
    print("   [✓] Track metadata and metrics")

    print("\n💡 Error Handling:")
    print("   - Validate token_ids before passing to model")
    print("   - Catch parsing errors and provide fallbacks")
    print("   - Log failures for debugging")
    print("   - Always return user-friendly error messages")

    print("\n💡 Performance Tips:")
    print("   - Reuse HarmonyPromptBuilder instances")
    print("   - Use extract_final_only=True when only final response needed")
    print("   - Monitor token counts to stay within context limits")
    print("   - Cache system/developer prompts when possible")

    print("\n✅ Complete workflow finished!")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "HARMONY FORMAT EXAMPLES - NEW API" + " " * 16 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nThis script demonstrates the new openai-harmony implementation.")
    print("All examples use mock responses for demonstration purposes.")
    print("\nIn production, connect to a real MLX model using InferenceEngine.")

    examples = [
        ("1. Basic Usage", example_1_basic_usage),
        ("2. Reasoning Levels", example_2_reasoning_levels),
        ("3. Multi-Turn Conversations", example_3_multi_turn_conversation),
        ("4. Presets Integration", example_4_presets_integration),
        ("5. Streaming with Channels", example_5_streaming_with_channels),
        ("6. Complete Workflow", example_6_complete_workflow),
    ]

    print("\n" + "=" * 70)
    print(" AVAILABLE EXAMPLES")
    print("=" * 70)
    for name, _ in examples:
        print(f"  {name}")
    print("=" * 70)

    # Run all examples
    try:
        for name, example_func in examples:
            example_func()

        print_section("Summary")
        print("All examples completed successfully! 🎉")
        print("\n✨ Key Improvements Over Old Implementation:")
        print("   1. Official openai-harmony package (better compatibility)")
        print("   2. Token-based prompt building (more accurate)")
        print("   3. StreamableParser for robust channel extraction")
        print("   4. Type-safe interfaces with clear contracts")
        print("   5. Better error handling and metadata tracking")
        print("   6. Integration with presets system")
        print("   7. Support for multi-channel streaming")
        print("   8. Production-ready with performance optimizations")

        print("\n📚 Educational Value:")
        print("   ✓ Learn Harmony format fundamentals")
        print("   ✓ Understand reasoning levels")
        print("   ✓ Master multi-turn conversations")
        print("   ✓ Integrate with presets effectively")
        print("   ✓ Handle streaming responses")
        print("   ✓ Build complete production workflows")

        print("\n📖 For More Information:")
        print("   - API Documentation: src/prompts/harmony_native.py")
        print("   - Presets Guide: src/prompts/presets.py")
        print("   - Integration Contracts: .context-kit/orchestration/harmony-replacement/")
        print("   - Tests: tests/unit/test_prompts/test_harmony_native.py")

        print("\n🚀 Next Steps:")
        print("   1. Run with real MLX model: python src/main.py")
        print("   2. Experiment with different reasoning levels")
        print("   3. Try different presets for various use cases")
        print("   4. Build your own custom conversations")
        print("   5. Integrate streaming for better UX")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
