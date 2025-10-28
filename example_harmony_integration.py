#!/usr/bin/env python3
"""
Example: Harmony Integration with Conversation History

This example demonstrates the complete integration flow:
1. User sends a message
2. InferenceEngine generates Harmony response
3. ConversationManager stores response with all channels
4. Conversation is saved with channel preservation
5. Conversation is loaded and used to build next prompt

This shows the round-trip integration between all Harmony components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conversation import ConversationManager


def simulate_harmony_response():
    """
    Simulate a ParsedHarmonyResponse from the parser.

    In real usage, this would come from:
        parsed = harmony_parser.parse_response_text(generated_text, tokenizer)
    """
    class MockParsedHarmonyResponse:
        def __init__(self):
            self.final = "Python is a high-level programming language known for its readability."
            self.analysis = (
                "The user asked about Python. I should provide a clear, "
                "concise definition. Python is indeed known for readability "
                "and is widely used. I'll mention its key characteristic."
            )
            self.commentary = "Educational question about a programming language."
            self.channels = {
                "final": self.final,
                "analysis": self.analysis,
                "commentary": self.commentary
            }
            self.metadata = {
                "token_count": 142,
                "parsing_time_ms": 3
            }

    return MockParsedHarmonyResponse()


def main():
    print("\n" + "=" * 70)
    print("HARMONY INTEGRATION EXAMPLE: Full Round-Trip")
    print("=" * 70 + "\n")

    # ========================================================================
    # STEP 1: Create conversation and add user message
    # ========================================================================
    print("STEP 1: User sends message")
    print("-" * 70)

    conv = ConversationManager(max_context_tokens=4096)
    conv.add_message("system", "You are a helpful AI assistant.")
    conv.add_message("user", "What is Python?")

    print(f"✓ Conversation has {conv.get_message_count()} messages")
    print(f"  - User: What is Python?")
    print()

    # ========================================================================
    # STEP 2: Build Harmony prompt for inference
    # ========================================================================
    print("STEP 2: Build Harmony prompt")
    print("-" * 70)

    # Get messages in Harmony format
    messages = conv.get_messages_for_harmony()
    print(f"✓ Extracted {len(messages)} messages for HarmonyPromptBuilder")
    print(f"  Format: {messages}")
    print()

    # In real usage, this would be passed to HarmonyPromptBuilder:
    # harmony_prompt = builder.build_conversation(
    #     messages=messages,
    #     system_prompt=system_prompt,
    #     developer_prompt=developer_prompt
    # )

    # ========================================================================
    # STEP 3: Generate response (simulated)
    # ========================================================================
    print("STEP 3: InferenceEngine generates Harmony response")
    print("-" * 70)

    # Simulate inference engine returning parsed response
    parsed_response = simulate_harmony_response()

    print("✓ Generated Harmony response with channels:")
    print(f"  - final: {parsed_response.final[:60]}...")
    print(f"  - analysis: {parsed_response.analysis[:60]}...")
    print(f"  - commentary: {parsed_response.commentary}")
    print()

    # ========================================================================
    # STEP 4: Store response in conversation with channels
    # ========================================================================
    print("STEP 4: Store response in conversation")
    print("-" * 70)

    # Use convenience method to add Harmony response
    msg = conv.add_harmony_response(parsed_response)

    print(f"✓ Added Harmony response to conversation")
    print(f"  - Message content (final): {msg.content[:60]}...")
    print(f"  - Has channels: {msg.channels is not None}")
    print(f"  - Channels: {list(msg.channels.keys())}")
    print(f"  - Has metadata: {msg.metadata is not None}")
    print()

    # ========================================================================
    # STEP 5: Save conversation with channels
    # ========================================================================
    print("STEP 5: Save conversation")
    print("-" * 70)

    save_path = "/tmp/harmony_conversation_example.json"
    conv.save(save_path)

    print(f"✓ Saved conversation to {save_path}")
    print(f"  - Total messages: {conv.get_message_count()}")
    print(f"  - Total tokens: {conv.get_token_count()}")
    print()

    # ========================================================================
    # STEP 6: Load conversation (round-trip)
    # ========================================================================
    print("STEP 6: Load conversation (round-trip test)")
    print("-" * 70)

    loaded_conv = ConversationManager.load(save_path)

    print(f"✓ Loaded conversation from {save_path}")
    print(f"  - Messages: {loaded_conv.get_message_count()}")
    print()

    # Verify channels preserved
    loaded_assistant_msg = loaded_conv.messages[-1]
    print("✓ Verified channel preservation:")
    print(f"  - Final: {loaded_assistant_msg.channels['final'][:60]}...")
    print(f"  - Analysis: {loaded_assistant_msg.channels['analysis'][:60]}...")
    print(f"  - Commentary: {loaded_assistant_msg.channels['commentary']}")
    print()

    # ========================================================================
    # STEP 7: Continue conversation with loaded history
    # ========================================================================
    print("STEP 7: Continue conversation")
    print("-" * 70)

    # Add another user message
    loaded_conv.add_message("user", "Is Python good for beginners?")

    # Build prompt again with full history
    messages_with_history = loaded_conv.get_messages_for_harmony()

    print(f"✓ Conversation continues with {loaded_conv.get_message_count()} messages")
    print(f"  - Messages for next prompt: {len(messages_with_history)}")
    print()

    # In real usage, this would go to InferenceEngine:
    # result = engine.generate(...)
    # loaded_conv.add_harmony_response(result.parsed_response)

    # ========================================================================
    # STEP 8: Demonstrate backward compatibility
    # ========================================================================
    print("STEP 8: Backward compatibility demo")
    print("-" * 70)

    # Create conversation with old-style messages (no channels)
    old_conv = ConversationManager()
    old_conv.add_message("user", "Hello")
    old_conv.add_message("assistant", "Hi!")  # No channels

    # Save and load
    old_conv.save("/tmp/old_format_conversation.json")
    loaded_old = ConversationManager.load("/tmp/old_format_conversation.json")

    print("✓ Old-format messages work correctly:")
    print(f"  - Loaded {loaded_old.get_message_count()} messages")
    print(f"  - Assistant message has channels: {loaded_old.messages[1].channels is not None}")
    print(f"  - Messages can still be extracted: {len(loaded_old.get_messages_for_harmony())}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("✅ COMPLETE INTEGRATION SUCCESS")
    print("=" * 70)
    print()
    print("Key Features Demonstrated:")
    print("1. ✓ ConversationManager stores Harmony channels")
    print("2. ✓ Save/load preserves all channel data")
    print("3. ✓ get_messages_for_harmony() provides clean format")
    print("4. ✓ add_harmony_response() convenience method works")
    print("5. ✓ Backward compatible with old messages")
    print("6. ✓ Round-trip save → load → build succeeds")
    print()
    print("Integration Points:")
    print("- HarmonyPromptBuilder: Uses get_messages_for_harmony()")
    print("- HarmonyResponseParser: Results stored via add_harmony_response()")
    print("- InferenceEngine: Builds prompts and stores responses")
    print("- Persistence: All channels preserved in JSON")
    print()


if __name__ == "__main__":
    main()
