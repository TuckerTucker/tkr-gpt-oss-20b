#!/usr/bin/env python3
"""
Comprehensive demo of GPT-OSS CLI Chat Wave 2 features.

This demo showcases:
- All 9 Wave 2 commands
- Enhanced display with colors and metrics
- Advanced input with auto-completion
- Streaming display (mock)
- Progress indicators
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.commands import CommandDispatcher, COMMANDS
from src.cli.display import (
    MessageFormatter,
    WelcomeDisplay,
    StatisticsDisplay,
    ProgressDisplay,
    StreamingDisplay,
    ConversationDisplay,
)
from src.cli.output import (
    console,
    print_separator,
    print_system_message,
    print_info,
    print_success,
    print_panel,
)
from src.conversation.history import ConversationManager
from src.conversation.persistence import ConversationPersistence


def demo_welcome():
    """Demo welcome banner"""
    print_separator()
    console.print("[bold cyan]GPT-OSS CLI Chat - Wave 2 Feature Demo[/bold cyan]", justify="center")
    print_separator()
    print_system_message("This demo showcases all Wave 2 CLI enhancements")
    print_separator()


def demo_commands():
    """Demo all commands"""
    print_info("DEMO 1: Command System")
    print_separator(char="·")

    # Setup
    conversation = ConversationManager(max_context_tokens=4096)
    persistence = ConversationPersistence()
    model_info = {
        "model_name": "phi-3-mini",
        "context_length": 4096,
        "memory_mb": 2500,
    }
    dispatcher = CommandDispatcher(
        conversation=conversation,
        persistence=persistence,
        model_info=model_info,
    )

    # Add some sample messages
    conversation.add_message("user", "Hello, how are you?")
    conversation.add_message("assistant", "I'm doing great! How can I help you today?")
    conversation.add_message("user", "Can you explain what a neural network is?")
    conversation.add_message(
        "assistant",
        "A neural network is a computational model inspired by biological neural networks. "
        "It consists of interconnected nodes (neurons) organized in layers that process information."
    )

    # Demo /help
    print_system_message("\n1. Testing /help command:")
    _, output = dispatcher.execute("/help")
    print(output)

    time.sleep(1)

    # Demo /stats
    print_system_message("\n2. Testing /stats command:")
    _, output = dispatcher.execute("/stats")
    print(output)

    time.sleep(1)

    # Demo /info
    print_system_message("\n3. Testing /info command:")
    _, output = dispatcher.execute("/info")
    print(output)

    time.sleep(1)

    # Demo /history
    print_system_message("\n4. Testing /history command:")
    _, output = dispatcher.execute("/history 2")
    print(output)

    time.sleep(1)

    # Demo /search
    print_system_message("\n5. Testing /search command:")
    _, output = dispatcher.execute("/search neural")
    print(output)

    time.sleep(1)

    # Demo /save
    print_system_message("\n6. Testing /save command:")
    _, output = dispatcher.execute("/save demo_conversation.json")
    print(output)

    time.sleep(1)

    # Demo /export
    print_system_message("\n7. Testing /export command:")
    _, output = dispatcher.execute("/export markdown demo_export.md")
    print(output)

    time.sleep(1)

    # Demo /clear
    print_system_message("\n8. Testing /clear command:")
    _, output = dispatcher.execute("/clear")
    print(output)

    print_success("All commands demonstrated successfully!")
    print_separator()


def demo_display():
    """Demo enhanced display features"""
    print_info("DEMO 2: Enhanced Display")
    print_separator(char="·")

    # Message formatter
    print_system_message("\n1. Message formatting with colors:")
    formatter = MessageFormatter(show_metadata=True)

    formatter.display_user_message("This is a user message")
    formatter.display_assistant_message(
        "This is an assistant message",
        metadata={
            "model": "phi-3-mini",
            "latency_ms": 245,
            "tokens": {"total_tokens": 42},
        }
    )
    formatter.display_system_message("This is a system message")

    time.sleep(1)

    # Statistics display
    print_system_message("\n2. Token usage statistics:")
    StatisticsDisplay.display_token_usage({
        "prompt_tokens": 150,
        "completion_tokens": 320,
        "total_tokens": 470,
    })

    time.sleep(1)

    # Performance metrics
    print_system_message("\n3. Performance metrics:")
    StatisticsDisplay.display_performance_metrics({
        "latency_ms": 245.7,
        "tokens_per_second": 67.3,
        "memory_used_mb": 2450,
        "cpu_percent": 42.5,
    })

    time.sleep(1)

    # Generation metrics
    print_system_message("\n4. Generation performance:")
    StatisticsDisplay.display_generation_metrics(
        tokens=150,
        duration_ms=2234,
        tokens_per_second=67.1
    )

    print_success("Display features demonstrated successfully!")
    print_separator()


def demo_progress():
    """Demo progress indicators"""
    print_info("DEMO 3: Progress Indicators")
    print_separator(char="·")

    # Loading spinner
    print_system_message("\n1. Loading spinner:")
    with ProgressDisplay.loading_spinner("Loading model...") as progress:
        task = progress.add_task("Loading model...", total=None)
        time.sleep(2)

    print_success("Model loaded!")

    time.sleep(1)

    # Progress bar
    print_system_message("\n2. Progress bar:")
    with ProgressDisplay.progress_bar("Processing documents") as progress:
        task = progress.add_task("Processing documents", total=100)
        for i in range(100):
            time.sleep(0.02)
            progress.update(task, advance=1)

    print_success("Processing complete!")
    print_separator()


def demo_streaming():
    """Demo streaming display"""
    print_info("DEMO 4: Streaming Display")
    print_separator(char="·")

    # Mock token stream
    def mock_token_stream():
        """Generate mock tokens"""
        text = "This is a demonstration of streaming text generation. "
        text += "Tokens appear one at a time, simulating real-time inference. "
        text += "The display shows live metrics including token count and speed."
        for word in text.split():
            yield word + " "
            time.sleep(0.05)

    print_system_message("\n1. Simple streaming:")
    streamer = StreamingDisplay(show_role=True)
    result = streamer.stream_tokens(mock_token_stream(), role="assistant")

    time.sleep(1)

    print_system_message("\n2. Streaming with live metrics:")
    streamer2 = StreamingDisplay(show_role=True)
    result2 = streamer2.stream_with_live_metrics(mock_token_stream(), role="assistant")

    print_success("Streaming demonstrated successfully!")
    print_separator()


def demo_conversation_display():
    """Demo conversation history display"""
    print_info("DEMO 5: Conversation Display")
    print_separator(char="·")

    # Create sample conversation
    conv = ConversationManager()
    conv.add_message("system", "You are a helpful AI assistant.")
    conv.add_message("user", "What is Python?")
    conv.add_message(
        "assistant",
        "Python is a high-level, interpreted programming language known for its "
        "simplicity and readability. It's widely used in web development, data science, "
        "machine learning, and automation."
    )
    conv.add_message("user", "Can you give me a simple example?")
    conv.add_message(
        "assistant",
        "Sure! Here's a simple Python program:\n\n"
        "```python\n"
        "print('Hello, World!')\n"
        "```\n\n"
        "This prints 'Hello, World!' to the console."
    )

    # Display conversation
    display = ConversationDisplay()
    display.display_history(
        [{"role": m.role, "content": m.content} for m in conv.messages]
    )

    time.sleep(1)

    # Display summary
    display.display_summary(
        [{"role": m.role, "content": m.content} for m in conv.messages]
    )

    print_success("Conversation display demonstrated successfully!")
    print_separator()


def demo_welcome_screen():
    """Demo welcome screen"""
    print_info("DEMO 6: Welcome Screen")
    print_separator(char="·")

    print_system_message("\n1. Welcome banner:")
    WelcomeDisplay.display_welcome(model_name="phi-3-mini")

    time.sleep(1)

    print_system_message("\n2. Loading indicator:")
    WelcomeDisplay.display_loading("Loading model from cache...")
    time.sleep(1)

    print_system_message("\n3. Ready message:")
    WelcomeDisplay.display_ready()

    print_success("Welcome screen demonstrated successfully!")
    print_separator()


def demo_command_list():
    """Demo command listing"""
    print_info("DEMO 7: Available Commands")
    print_separator(char="·")

    lines = ["All Wave 2 commands:\n"]
    for cmd_name, cmd in sorted(COMMANDS.items()):
        lines.append(f"  [cyan]{cmd.name:15}[/cyan] {cmd.description}")

    console.print("\n".join(lines))

    print_separator()


def main():
    """Run all demos"""
    demo_welcome()

    # Run demos
    demos = [
        ("Command System", demo_commands),
        ("Enhanced Display", demo_display),
        ("Progress Indicators", demo_progress),
        ("Streaming Display", demo_streaming),
        ("Conversation Display", demo_conversation_display),
        ("Welcome Screen", demo_welcome_screen),
        ("Command List", demo_command_list),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            console.print(f"\n[bold yellow]Demo {i}/{len(demos)}: {name}[/bold yellow]")
            demo_func()
            time.sleep(0.5)
        except KeyboardInterrupt:
            print_system_message("\nDemo interrupted by user")
            break
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()

    # Final message
    print_separator()
    console.print("[bold green]✓ All demos completed![/bold green]", justify="center")
    print_separator()

    print_panel(
        "Wave 2 features demonstrated:\n\n"
        "✓ 9 fully-functional commands\n"
        "✓ Colored output for all message types\n"
        "✓ Progress indicators and spinners\n"
        "✓ Streaming display with live metrics\n"
        "✓ Token count and performance tracking\n"
        "✓ Conversation history and statistics\n"
        "✓ Command auto-completion ready\n"
        "✓ Multi-line input support ready",
        title="Wave 2 Complete",
        style="green"
    )


if __name__ == "__main__":
    main()
