#!/usr/bin/env python3
"""
Test script for Harmony conversation integration.

This script tests:
1. Backward compatibility with old message format
2. Round-trip save/load with channels
3. Integration with HarmonyPromptBuilder
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from conversation import ConversationManager, Message


def test_backward_compatibility():
    """Test that old message format still works."""
    print("=" * 60)
    print("TEST 1: Backward Compatibility")
    print("=" * 60)

    # Create conversation with old-style messages (no channels)
    conv = ConversationManager()

    msg1 = conv.add_message("user", "Hello!")
    print(f"✓ Added user message: {msg1.content}")
    assert msg1.role == "user"
    assert msg1.content == "Hello!"
    assert msg1.channels is None
    assert msg1.metadata is None

    msg2 = conv.add_message("assistant", "Hi there!")
    print(f"✓ Added assistant message: {msg2.content}")
    assert msg2.role == "assistant"
    assert msg2.content == "Hi there!"
    assert msg2.channels is None

    # Test to_dict
    data = conv.to_dict()
    print(f"✓ Converted to dict: {len(data['messages'])} messages")
    assert len(data["messages"]) == 2

    # Old messages should not have 'channels' key in dict
    for msg_dict in data["messages"]:
        assert "channels" not in msg_dict or msg_dict["channels"] is None

    print("✓ Old message format works correctly (no channel clutter)\n")
    return True


def test_harmony_channels():
    """Test Harmony channel storage."""
    print("=" * 60)
    print("TEST 2: Harmony Channel Storage")
    print("=" * 60)

    conv = ConversationManager()

    # Add user message
    conv.add_message("user", "What is 2+2?")
    print("✓ Added user message")

    # Add assistant message with channels
    channels = {
        "final": "The answer is 4.",
        "analysis": "To solve 2+2, I add the two numbers together.",
        "commentary": "This is a simple arithmetic problem."
    }

    msg = conv.add_message(
        "assistant",
        "The answer is 4.",
        channels=channels,
        metadata={"has_harmony_channels": True}
    )

    print(f"✓ Added Harmony response with {len(channels)} channels")
    assert msg.channels == channels
    assert msg.metadata["has_harmony_channels"] is True

    # Verify channel access
    assert msg.channels["final"] == "The answer is 4."
    assert msg.channels["analysis"] == "To solve 2+2, I add the two numbers together."
    assert msg.channels["commentary"] == "This is a simple arithmetic problem."
    print("✓ All channels accessible\n")

    return conv


def test_save_load_roundtrip():
    """Test save/load preserves channels."""
    print("=" * 60)
    print("TEST 3: Save/Load Round-trip")
    print("=" * 60)

    # Create conversation with channels
    conv1 = ConversationManager()
    conv1.add_message("user", "Hello!")
    conv1.add_message(
        "assistant",
        "Hi there!",
        channels={
            "final": "Hi there!",
            "analysis": "User greeted me, responding warmly.",
            "commentary": "Standard greeting exchange."
        }
    )

    print(f"✓ Created conversation with {conv1.get_message_count()} messages")

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        conv1.save(temp_path)
        print(f"✓ Saved to {temp_path}")

        # Verify JSON structure
        with open(temp_path, 'r') as f:
            data = json.load(f)

        print(f"✓ JSON contains {len(data['messages'])} messages")

        # Check second message has channels
        msg2_data = data['messages'][1]
        assert "channels" in msg2_data
        assert msg2_data["channels"]["final"] == "Hi there!"
        assert msg2_data["channels"]["analysis"] == "User greeted me, responding warmly."
        print("✓ Channels preserved in JSON")

        # Load conversation
        conv2 = ConversationManager.load(temp_path)
        print(f"✓ Loaded conversation with {conv2.get_message_count()} messages")

        # Verify channels preserved
        loaded_msg = conv2.messages[1]
        assert loaded_msg.channels is not None
        assert loaded_msg.channels["final"] == "Hi there!"
        assert loaded_msg.channels["analysis"] == "User greeted me, responding warmly."
        assert loaded_msg.channels["commentary"] == "Standard greeting exchange."
        print("✓ All channels preserved after load\n")

        return True

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_get_messages_for_harmony():
    """Test message extraction for HarmonyPromptBuilder."""
    print("=" * 60)
    print("TEST 4: HarmonyPromptBuilder Integration")
    print("=" * 60)

    conv = ConversationManager()
    conv.add_message("system", "You are helpful.")
    conv.add_message("user", "What is AI?")
    conv.add_message(
        "assistant",
        "AI is artificial intelligence.",
        channels={
            "final": "AI is artificial intelligence.",
            "analysis": "User asking about AI definition...",
            "commentary": "Educational question."
        }
    )

    # Get messages for Harmony
    messages = conv.get_messages_for_harmony()
    print(f"✓ Extracted {len(messages)} messages for Harmony")

    # Verify format
    assert len(messages) == 3
    assert all("role" in msg and "content" in msg for msg in messages)

    # Verify content uses final channel
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "AI is artificial intelligence."
    print("✓ Messages in correct format for HarmonyPromptBuilder")

    # Verify only role and content keys (no channels in output)
    for msg in messages:
        assert set(msg.keys()) == {"role", "content"}
    print("✓ Clean message format (channels not included)\n")

    return True


def test_mixed_messages():
    """Test conversation with both Harmony and non-Harmony messages."""
    print("=" * 60)
    print("TEST 5: Mixed Harmony/Non-Harmony Messages")
    print("=" * 60)

    conv = ConversationManager()

    # Old-style message
    conv.add_message("user", "Hello!")

    # Harmony message
    conv.add_message(
        "assistant",
        "Hi!",
        channels={"final": "Hi!", "analysis": "Greeting"}
    )

    # Another old-style message
    conv.add_message("user", "How are you?")

    # Another Harmony message
    conv.add_message(
        "assistant",
        "I'm good!",
        channels={"final": "I'm good!", "analysis": "Positive response"}
    )

    print(f"✓ Created mixed conversation with {conv.get_message_count()} messages")

    # Verify to_dict handles both
    data = conv.to_dict()
    assert "channels" not in data["messages"][0]  # User message
    assert "channels" in data["messages"][1]  # Harmony response
    assert "channels" not in data["messages"][2]  # User message
    assert "channels" in data["messages"][3]  # Harmony response

    print("✓ Mixed messages handled correctly")

    # Verify get_messages_for_harmony works
    messages = conv.get_messages_for_harmony()
    assert len(messages) == 4
    assert all(msg["content"] for msg in messages)
    print("✓ Mixed messages extract correctly for Harmony\n")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HARMONY CONVERSATION INTEGRATION TESTS")
    print("=" * 60 + "\n")

    try:
        test_backward_compatibility()
        test_harmony_channels()
        test_save_load_roundtrip()
        test_get_messages_for_harmony()
        test_mixed_messages()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print("- Backward compatibility: ✓")
        print("- Harmony channel storage: ✓")
        print("- Save/load round-trip: ✓")
        print("- HarmonyPromptBuilder integration: ✓")
        print("- Mixed message handling: ✓")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
