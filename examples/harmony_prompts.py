#!/usr/bin/env python3
"""
Harmony Format Examples

This script demonstrates how to use the Harmony multi-channel format
for enhanced reasoning, debugging, and transparency.

Examples included:
1. Summarization with reasoning display
2. Extraction with citations
3. Chat with reasoning traces
4. Developer role with analysis access
5. Accessing and using reasoning traces

Run: python examples/harmony_prompts.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.inference_config import InferenceConfig, ReasoningLevel
from src.config.model_config import ModelConfig
from src.models.loader import ModelLoader
from src.inference.engine import InferenceEngine
from src.sampling.params import SamplingParams
from src.prompts.harmony import HarmonyEncoder, HarmonyMessage, Role, Channel
from src.prompts.harmony_channels import (
    extract_channel,
    extract_all_channels,
    format_reasoning_trace
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def example_1_summarization_with_reasoning():
    """
    Example 1: Summarization with Reasoning Display

    Shows how to use Harmony to see the model's reasoning process
    while performing a summarization task.
    """
    print_section("Example 1: Summarization with Reasoning")

    # Configure with reasoning capture enabled
    config = InferenceConfig(
        use_harmony_format=True,
        reasoning_level=ReasoningLevel.MEDIUM,
        capture_reasoning=True,  # Store reasoning in result
        show_reasoning=False,    # We'll display it manually
        temperature=0.7,
        max_tokens=512
    )

    print("Configuration:")
    print(f"  - Harmony enabled: {config.use_harmony_format}")
    print(f"  - Reasoning level: {config.reasoning_level.value}")
    print(f"  - Capture reasoning: {config.capture_reasoning}")

    # Sample text to summarize
    article = """
    Artificial intelligence has made remarkable progress in recent years,
    particularly in natural language processing. Large language models can
    now generate coherent text, answer questions, and even write code.
    However, these models also face challenges including bias, hallucinations,
    and high computational costs. Researchers are working on making models
    more efficient, reliable, and transparent.
    """

    print("\nArticle to summarize:")
    print(article.strip())

    # Mock generation for demo (in real usage, this would call the model)
    print("\n[Note: This is a demo with mock responses. In production, connect to a real model.]")

    # Simulated Harmony response from model
    mock_response = """
<|start|>assistant<|channel|>analysis<|message|>
Let me analyze this article step by step:
1. Main topic: Recent AI progress, specifically in NLP
2. Key achievements: Text generation, Q&A, code writing
3. Key challenges: Bias, hallucinations, computational cost
4. Current work: Efficiency, reliability, transparency improvements

I should summarize these four main points concisely.
<|end|>
<|start|>assistant<|channel|>final<|message|>
AI, especially in natural language processing, has advanced significantly with large language
models capable of text generation, question answering, and coding. However, challenges remain
including bias, hallucinations, and high computational costs. Researchers are focusing on
improving efficiency, reliability, and transparency.
<|end|>
"""

    # Parse the response using HarmonyEncoder
    encoder = HarmonyEncoder()
    parsed = encoder.parse_response(mock_response)

    print("\n--- Final Summary (User-Facing) ---")
    print(parsed.final)

    print("\n--- Reasoning Trace (Developer View) ---")
    if parsed.analysis:
        formatted = format_reasoning_trace(parsed.analysis, max_length=500)
        print(formatted)
    else:
        print("No reasoning captured")

    print("\n--- All Channels ---")
    channels = extract_all_channels(mock_response)
    for channel_name, content in channels.items():
        print(f"\n[{channel_name.upper()}]")
        print(content[:200] + "..." if len(content) > 200 else content)


def example_2_extraction_with_citations():
    """
    Example 2: Data Extraction with Low Reasoning

    Demonstrates using low reasoning level for fast, straightforward
    extraction tasks where heavy analysis isn't needed.
    """
    print_section("Example 2: Data Extraction with Low Reasoning")

    # Configure for fast extraction
    config = InferenceConfig(
        use_harmony_format=True,
        reasoning_level=ReasoningLevel.LOW,  # Minimal reasoning for speed
        capture_reasoning=True,
        temperature=0.3,  # Lower temperature for factual extraction
        max_tokens=256
    )

    print("Configuration:")
    print(f"  - Reasoning level: {config.reasoning_level.value} (fast)")
    print(f"  - Temperature: {config.temperature} (focused)")

    # Sample text with structured data
    text = """
    Contact Information:
    Name: Dr. Alice Johnson
    Email: alice.johnson@example.com
    Phone: +1-555-0123
    Department: Research & Development
    """

    print("\nText to extract from:")
    print(text.strip())

    # Mock extraction response
    mock_response = """
<|start|>assistant<|channel|>analysis<|message|>
Extract structured data: name, email, phone, department.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Extracted Information:
- Name: Dr. Alice Johnson
- Email: alice.johnson@example.com
- Phone: +1-555-0123
- Department: Research & Development
<|end|>
"""

    encoder = HarmonyEncoder()
    parsed = encoder.parse_response(mock_response)

    print("\n--- Extracted Data ---")
    print(parsed.final)

    print("\n--- Reasoning (Minimal for Low Level) ---")
    if parsed.analysis:
        print(f"Analysis: {parsed.analysis}")
        print("\nNote: Low reasoning level produces minimal analysis for speed.")
    else:
        print("No reasoning captured")


def example_3_chat_with_reasoning():
    """
    Example 3: Chat with High Reasoning

    Shows how high reasoning level provides detailed chain-of-thought
    for complex questions.
    """
    print_section("Example 3: Chat with High Reasoning")

    # Configure for maximum reasoning
    config = InferenceConfig(
        use_harmony_format=True,
        reasoning_level=ReasoningLevel.HIGH,  # Maximum reasoning
        capture_reasoning=True,
        show_reasoning=True,
        temperature=0.8,
        max_tokens=1024
    )

    print("Configuration:")
    print(f"  - Reasoning level: {config.reasoning_level.value} (thorough)")
    print(f"  - Max tokens: {config.max_tokens} (longer responses)")

    question = "What are the ethical implications of using AI in hiring decisions?"

    print(f"\nQuestion: {question}")

    # Mock response with extensive reasoning
    mock_response = """
<|start|>assistant<|channel|>analysis<|message|>
This is a complex ethical question requiring careful analysis:

1. Key stakeholders: Job applicants, employers, society
2. Potential benefits: Reduced bias, efficiency, consistency
3. Potential harms: Algorithmic bias, lack of transparency, discrimination
4. Ethical frameworks: Fairness, accountability, transparency
5. Real-world considerations: Legal compliance, public trust

I should present a balanced view covering both opportunities and concerns,
emphasizing the importance of responsible AI deployment.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Using AI in hiring raises several important ethical considerations:

**Potential Benefits:**
- Can reduce human bias and increase objectivity
- Enables consistent evaluation across all candidates
- Improves efficiency in processing large applicant pools

**Ethical Concerns:**
- Risk of encoding historical biases from training data
- Lack of transparency in decision-making ("black box" problem)
- Potential for discriminatory outcomes against protected groups
- Difficulty in providing meaningful explanations to rejected candidates

**Key Requirements:**
- Regular audits for bias and fairness
- Transparency in how AI systems make decisions
- Human oversight and appeal processes
- Compliance with anti-discrimination laws

The ethical use of AI in hiring requires careful design, ongoing monitoring,
and commitment to fairness and transparency.
<|end|>
"""

    encoder = HarmonyEncoder()
    parsed = encoder.parse_response(mock_response)

    print("\n--- Response ---")
    print(parsed.final)

    print("\n--- Detailed Reasoning Trace ---")
    if parsed.analysis:
        print(parsed.analysis)
        print("\nNote: High reasoning level provides extensive analysis, improving quality.")
    else:
        print("No reasoning captured")


def example_4_developer_role():
    """
    Example 4: Developer Role with Analysis Access

    Shows how developers can access reasoning traces programmatically
    for debugging and quality improvement.
    """
    print_section("Example 4: Developer Role - Accessing Reasoning")

    print("Scenario: A developer debugging unexpected model behavior\n")

    # Simulate multiple generations with different reasoning levels
    test_cases = [
        ("low", "What is 2+2?"),
        ("medium", "Explain photosynthesis"),
        ("high", "Design a scalable microservices architecture")
    ]

    for level, prompt in test_cases:
        print(f"\n--- Test: {prompt} (Reasoning: {level}) ---")

        # Mock response
        mock_responses = {
            "low": """
<|start|>assistant<|channel|>analysis<|message|>
Simple arithmetic.
<|end|>
<|start|>assistant<|channel|>final<|message|>
2 + 2 = 4
<|end|>
""",
            "medium": """
<|start|>assistant<|channel|>analysis<|message|>
Photosynthesis is a biological process. Key points: light energy, CO2, water, oxygen, glucose.
Need to explain the process clearly for general audience.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Photosynthesis is the process by which plants convert light energy into chemical energy.
Plants use sunlight, carbon dioxide, and water to produce glucose (sugar) and oxygen.
The oxygen is released as a byproduct, while the glucose provides energy for the plant.
<|end|>
""",
            "high": """
<|start|>assistant<|channel|>analysis<|message|>
Complex architecture question requiring systematic approach:
1. Define microservices principles
2. Consider scalability requirements
3. Address common challenges (service discovery, data consistency, communication)
4. Provide concrete recommendations

Should structure response as: principles, architecture components, best practices.
<|end|>
<|start|>assistant<|channel|>commentary<|message|>
Drawing on software architecture knowledge base.
Considering industry best practices.
<|end|>
<|start|>assistant<|channel|>final<|message|>
A scalable microservices architecture should include:

**Core Principles:**
- Single responsibility per service
- Independent deployment and scaling
- Decentralized data management

**Key Components:**
- API Gateway for routing and authentication
- Service registry for discovery
- Message queue for async communication
- Distributed caching layer
- Centralized logging and monitoring

**Best Practices:**
- Use containerization (Docker/Kubernetes)
- Implement circuit breakers for fault tolerance
- Design for eventual consistency
- Employ automated testing and CI/CD
<|end|>
"""
        }

        encoder = HarmonyEncoder()
        parsed = encoder.parse_response(mock_responses[level])

        # Developer analysis
        print(f"Response length: {len(parsed.final)} chars")
        print(f"Reasoning length: {len(parsed.analysis or '')} chars")
        print(f"Has commentary: {parsed.commentary is not None}")

        # Extract and format reasoning for logs
        if parsed.analysis:
            formatted = format_reasoning_trace(parsed.analysis, max_length=150)
            print(f"Reasoning preview: {formatted}")


def example_5_accessing_reasoning_traces():
    """
    Example 5: Programmatic Access to Reasoning

    Demonstrates the full API for working with Harmony responses.
    """
    print_section("Example 5: Programmatic Reasoning Access")

    mock_response = """
<|start|>assistant<|channel|>analysis<|message|>
User asks about quantum computing. Complex topic requiring:
1. Simple definition
2. Key concepts (qubits, superposition, entanglement)
3. Practical applications
4. Current limitations
<|end|>
<|start|>assistant<|channel|>commentary<|message|>
Accessing knowledge base on quantum computing.
Simplifying technical concepts for general audience.
<|end|>
<|start|>assistant<|channel|>final<|message|>
Quantum computing uses quantum mechanical phenomena to perform calculations.
Unlike classical bits (0 or 1), quantum bits (qubits) can exist in superposition,
representing multiple states simultaneously. This enables quantum computers to solve
certain problems exponentially faster than classical computers, particularly in
cryptography, optimization, and drug discovery. However, quantum computers are still
experimental and face challenges like error rates and maintaining quantum coherence.
<|end|>
"""

    print("Methods for accessing Harmony channels:\n")

    # Method 1: Using HarmonyEncoder
    print("1. HarmonyEncoder.parse_response()")
    encoder = HarmonyEncoder()
    parsed = encoder.parse_response(mock_response)

    print(f"   - final: {len(parsed.final)} chars")
    print(f"   - analysis: {len(parsed.analysis or '')} chars")
    print(f"   - commentary: {len(parsed.commentary or '')} chars")
    print(f"   - raw: {len(parsed.raw)} chars")

    # Method 2: Using channel extraction utilities
    print("\n2. extract_channel() utility")
    final = extract_channel(mock_response, "final")
    analysis = extract_channel(mock_response, "analysis")
    commentary = extract_channel(mock_response, "commentary")

    print(f"   - final: {final[:50]}...")
    print(f"   - analysis: {analysis[:50] if analysis else 'None'}...")
    print(f"   - commentary: {commentary[:50] if commentary else 'None'}...")

    # Method 3: Extract all channels at once
    print("\n3. extract_all_channels() utility")
    channels = extract_all_channels(mock_response)

    print(f"   - Found {len(channels)} channels: {list(channels.keys())}")
    for name, content in channels.items():
        print(f"   - {name}: {len(content)} chars")

    # Method 4: Format reasoning for display
    print("\n4. format_reasoning_trace() for display")
    if analysis:
        formatted = format_reasoning_trace(analysis, max_length=200)
        print(f"   Formatted: {formatted}")

    # Method 5: Validate Harmony format
    print("\n5. Validating Harmony format")
    is_valid = encoder.validate_format(mock_response)
    print(f"   Valid Harmony format: {is_valid}")

    # Show the complete API
    print("\n--- Complete API Usage Example ---")
    print("""
# In your application:
from src.prompts.harmony import HarmonyEncoder
from src.prompts.harmony_channels import extract_channel, format_reasoning_trace

encoder = HarmonyEncoder()
parsed = encoder.parse_response(model_output)

# For end users (SAFE)
display_to_user(parsed.final)

# For developers (DEBUG)
if parsed.analysis:
    log_reasoning(parsed.analysis)

# For tool usage tracking
if parsed.commentary:
    log_actions(parsed.commentary)

# Formatted for display
if config.show_reasoning and parsed.analysis:
    formatted = format_reasoning_trace(parsed.analysis, max_length=500)
    print(f"[Reasoning: {formatted}]")
""")


def example_6_encoding_messages():
    """
    Example 6: Encoding Messages with HarmonyEncoder

    Shows how to build Harmony-formatted prompts programmatically.
    """
    print_section("Example 6: Encoding Messages")

    encoder = HarmonyEncoder()

    # Build a conversation
    messages = [
        HarmonyMessage(Role.SYSTEM, "You are a helpful AI assistant."),
        HarmonyMessage(Role.USER, "What is machine learning?"),
        HarmonyMessage(
            Role.ASSISTANT,
            "Let me explain this step by step...",
            channel=Channel.ANALYSIS
        ),
        HarmonyMessage(
            Role.ASSISTANT,
            "Machine learning is a subset of artificial intelligence...",
            channel=Channel.FINAL
        ),
        HarmonyMessage(Role.USER, "Can you give me an example?")
    ]

    print("Encoding conversation with multiple messages and channels:\n")

    # Encode the conversation
    prompt = encoder.encode_conversation(messages, include_generation_prompt=True)

    print("Encoded prompt:")
    print(prompt)

    # Show structure
    print("\n--- Structure Analysis ---")
    print(f"Total length: {len(prompt)} characters")
    print(f"Number of messages: {len(messages)}")
    print(f"Includes generation prompt: Yes")
    print(f"Ready for model input: Yes")

    # Validate the encoded format
    is_valid = encoder.validate_format(prompt)
    print(f"Valid Harmony format: {is_valid}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "HARMONY FORMAT EXAMPLES" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nThis script demonstrates Harmony multi-channel format usage.")
    print("All examples use mock responses for demonstration purposes.")
    print("\nIn production, connect to a real model using ModelLoader and InferenceEngine.")

    # Run all examples
    try:
        example_1_summarization_with_reasoning()
        example_2_extraction_with_citations()
        example_3_chat_with_reasoning()
        example_4_developer_role()
        example_5_accessing_reasoning_traces()
        example_6_encoding_messages()

        print_section("Summary")
        print("All examples completed successfully!")
        print("\nKey takeaways:")
        print("  1. Use appropriate reasoning levels for different tasks")
        print("  2. Capture reasoning for debugging and quality improvement")
        print("  3. Always show parsed.final to end users (never parsed.analysis)")
        print("  4. Use channel extraction utilities for flexible access")
        print("  5. Format reasoning traces for readable display")
        print("\nFor more information:")
        print("  - User Guide: docs/harmony_integration.md")
        print("  - Migration Guide: docs/migration_to_harmony.md")
        print("  - API Reference: See docstrings in src/prompts/harmony.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
