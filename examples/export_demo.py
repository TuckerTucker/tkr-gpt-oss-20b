#!/usr/bin/env python3
"""
Example: Exporting Conversations

This script demonstrates the export functionality for conversations,
showing how to export to markdown, JSON, and text formats.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation import (
    ConversationManager,
    export_to_markdown,
    export_to_json,
    export_to_text,
    export_conversation,
)


def create_sample_conversation():
    """Create a sample conversation for demonstration."""
    conv = ConversationManager(
        max_context_tokens=2048,
        metadata={
            "model": "gpt-oss-20b",
            "session": "demo",
            "user": "example_user"
        }
    )

    # System prompt
    conv.add_message(
        "system",
        "You are a helpful AI assistant specialized in programming. "
        "Provide clear, concise answers with code examples when appropriate."
    )

    # Example conversation about Python
    conv.add_message(
        "user",
        "What is Python and why is it popular?"
    )

    conv.add_message(
        "assistant",
        "Python is a high-level, interpreted programming language known for its "
        "simplicity and readability. It's popular because:\n\n"
        "1. **Easy to Learn**: Clear syntax makes it beginner-friendly\n"
        "2. **Versatile**: Used in web development, data science, AI, automation\n"
        "3. **Rich Ecosystem**: Extensive libraries and frameworks\n"
        "4. **Strong Community**: Large, active community support\n\n"
        "Example of Python's simplicity:\n"
        "```python\n"
        "# Print hello world\n"
        "print('Hello, World!')\n"
        "```"
    )

    conv.add_message(
        "user",
        "Can you show me a more complex example?"
    )

    conv.add_message(
        "assistant",
        "Here's a practical example using Python for data processing:\n\n"
        "```python\n"
        "# Read CSV and calculate statistics\n"
        "import pandas as pd\n\n"
        "# Load data\n"
        "df = pd.read_csv('data.csv')\n\n"
        "# Calculate summary statistics\n"
        "stats = df.describe()\n"
        "print(stats)\n\n"
        "# Filter data\n"
        "filtered = df[df['age'] > 25]\n"
        "print(f'Records with age > 25: {len(filtered)}')\n"
        "```\n\n"
        "This demonstrates Python's power for data analysis with minimal code."
    )

    return conv


def main():
    """Demonstrate export functionality."""
    print("=== Conversation Export Demo ===\n")

    # Create sample conversation
    print("Creating sample conversation...")
    conv = create_sample_conversation()
    print(f"Created conversation with {conv.get_message_count()} messages\n")

    # Create output directory
    output_dir = Path(__file__).parent / "exported_conversations"
    output_dir.mkdir(exist_ok=True)

    # Export to Markdown
    print("1. Exporting to Markdown...")
    md_file = output_dir / "sample_conversation.md"
    export_to_markdown(conv, str(md_file), include_stats=True)
    print(f"   ✓ Exported to: {md_file}")
    print(f"   Size: {md_file.stat().st_size} bytes\n")

    # Export to JSON
    print("2. Exporting to JSON...")
    json_file = output_dir / "sample_conversation.json"
    export_to_json(conv, str(json_file), pretty=True)
    print(f"   ✓ Exported to: {json_file}")
    print(f"   Size: {json_file.stat().st_size} bytes\n")

    # Export to Text
    print("3. Exporting to plain text...")
    txt_file = output_dir / "sample_conversation.txt"
    export_to_text(conv, str(txt_file))
    print(f"   ✓ Exported to: {txt_file}")
    print(f"   Size: {txt_file.stat().st_size} bytes\n")

    # Auto-detect format
    print("4. Using auto-detect export...")
    auto_file = output_dir / "auto_detected.md"
    export_conversation(conv, str(auto_file))  # Auto-detects markdown from .md extension
    print(f"   ✓ Auto-detected format and exported to: {auto_file}\n")

    # Export without metadata
    print("5. Exporting markdown without metadata...")
    no_meta_file = output_dir / "no_metadata.md"
    export_to_markdown(conv, str(no_meta_file), include_metadata=False)
    print(f"   ✓ Exported minimal version to: {no_meta_file}\n")

    # Export compact JSON
    print("6. Exporting compact JSON...")
    compact_file = output_dir / "compact.json"
    export_to_json(conv, str(compact_file), pretty=False)
    print(f"   ✓ Exported compact JSON to: {compact_file}")
    print(f"   Size: {compact_file.stat().st_size} bytes (vs {json_file.stat().st_size} pretty)\n")

    print("✓ All exports complete!")
    print(f"\nView exported files in: {output_dir}")


if __name__ == "__main__":
    main()
