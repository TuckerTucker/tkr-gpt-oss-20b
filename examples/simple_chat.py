#!/usr/bin/env python3
"""
Simple Chat Example - Minimal GPT-OSS CLI Chat Demo

This example demonstrates the basic CLI functionality without requiring
the full inference stack. Perfect for testing the CLI interface in Wave 1.

Usage:
    python examples/simple_chat.py

Features:
    - Interactive REPL with command support
    - Colorized output
    - Command auto-completion
    - Mock responses (real inference in Wave 2)

Commands:
    /help    - Show available commands
    /quit    - Exit the program
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cli import REPL, print_system_message, print_separator


def main():
    """
    Run simple chat example.

    This demonstrates:
    1. Creating a REPL instance
    2. Running the interactive loop
    3. Using mock responses (no real model required)
    """

    # Display welcome message
    print_separator()
    print_system_message("GPT-OSS Simple Chat Example")
    print_system_message("Wave 1: CLI Interface Demo")
    print_separator()

    print_system_message("This example demonstrates the CLI without real inference.")
    print_system_message("Try typing 'hello', 'test', or any message!")
    print_system_message("Commands: /help, /quit")
    print_separator()

    # Create REPL instance (no conversation or engine needed for Wave 1)
    repl = REPL(
        conversation=None,  # Will be added in Wave 2
        engine=None,        # Will be added in Wave 2
        config=None,        # Will be added in Wave 2
    )

    # Run interactive loop
    try:
        repl.run()
    except KeyboardInterrupt:
        print()
        print_system_message("Interrupted. Goodbye!")
    except Exception as e:
        print_system_message(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
