"""
CLI module for GPT-OSS Chat.

Provides command-line interface components including:
- REPL: Interactive Read-Eval-Print Loop
- Commands: Command parsing and execution
- Output: Colorized output formatting
- Input: Enhanced input with auto-completion
- Display: Message and UI formatting
"""

from .repl import REPL, run_async_repl
from .commands import CommandParser, CommandDispatcher, COMMANDS
from .output import (
    console,
    format_user_message,
    format_assistant_message,
    format_system_message,
    format_error_message,
    print_user_message,
    print_assistant_message,
    print_system_message,
    print_error,
    print_info,
    print_success,
    print_panel,
    print_separator,
)
from .input import create_input_handler, EnhancedInput, SimpleInput
from .display import (
    MessageFormatter,
    ConversationDisplay,
    WelcomeDisplay,
    StatisticsDisplay,
)

__all__ = [
    # Main REPL
    "REPL",
    "run_async_repl",

    # Commands
    "CommandParser",
    "CommandDispatcher",
    "COMMANDS",

    # Output
    "console",
    "format_user_message",
    "format_assistant_message",
    "format_system_message",
    "format_error_message",
    "print_user_message",
    "print_assistant_message",
    "print_system_message",
    "print_error",
    "print_info",
    "print_success",
    "print_panel",
    "print_separator",

    # Input
    "create_input_handler",
    "EnhancedInput",
    "SimpleInput",

    # Display
    "MessageFormatter",
    "ConversationDisplay",
    "WelcomeDisplay",
    "StatisticsDisplay",
]

__version__ = "0.1.0"
