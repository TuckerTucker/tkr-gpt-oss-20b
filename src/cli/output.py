"""
Output formatting with colorized text using rich library.

Provides formatted output for user/assistant messages, system messages,
and other CLI elements.
"""

from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


# Initialize rich console
console = Console()


def format_user_message(text: str) -> str:
    """
    Format user message with color.

    Args:
        text: User message text

    Returns:
        Formatted string with rich markup
    """
    return f"[cyan]User:[/cyan] {text}"


def format_assistant_message(text: str) -> str:
    """
    Format assistant message with color.

    Args:
        text: Assistant message text

    Returns:
        Formatted string with rich markup
    """
    return f"[green]Assistant:[/green] {text}"


def format_system_message(text: str) -> str:
    """
    Format system message.

    Args:
        text: System message text

    Returns:
        Formatted string with rich markup
    """
    return f"[yellow]{text}[/yellow]"


def format_error_message(text: str) -> str:
    """
    Format error message.

    Args:
        text: Error message text

    Returns:
        Formatted string with rich markup
    """
    return f"[red]Error:[/red] {text}"


def format_info_message(text: str) -> str:
    """
    Format informational message.

    Args:
        text: Info message text

    Returns:
        Formatted string with rich markup
    """
    return f"[blue]ℹ[/blue] {text}"


def format_success_message(text: str) -> str:
    """
    Format success message.

    Args:
        text: Success message text

    Returns:
        Formatted string with rich markup
    """
    return f"[green]✓[/green] {text}"


def print_user_message(text: str) -> None:
    """
    Print formatted user message.

    Args:
        text: User message text
    """
    console.print(format_user_message(text))


def print_assistant_message(text: str) -> None:
    """
    Print formatted assistant message.

    Args:
        text: Assistant message text
    """
    console.print(format_assistant_message(text))


def print_system_message(text: str) -> None:
    """
    Print formatted system message.

    Args:
        text: System message text
    """
    console.print(format_system_message(text))


def print_error(text: str) -> None:
    """
    Print formatted error message.

    Args:
        text: Error message text
    """
    console.print(format_error_message(text))


def print_info(text: str) -> None:
    """
    Print formatted info message.

    Args:
        text: Info message text
    """
    console.print(format_info_message(text))


def print_success(text: str) -> None:
    """
    Print formatted success message.

    Args:
        text: Success message text
    """
    console.print(format_success_message(text))


def print_panel(content: str, title: Optional[str] = None, style: str = "cyan") -> None:
    """
    Print content in a rich panel.

    Args:
        content: Panel content
        title: Optional panel title
        style: Panel border style/color
    """
    console.print(Panel(content, title=title, border_style=style))


def print_markdown(text: str) -> None:
    """
    Print markdown-formatted text.

    Args:
        text: Markdown text
    """
    md = Markdown(text)
    console.print(md)


def print_separator(char: str = "─", width: Optional[int] = None) -> None:
    """
    Print a separator line.

    Args:
        char: Character to use for separator
        width: Optional width (defaults to console width)
    """
    if width is None:
        width = console.width
    console.print(char * width, style="dim")


def clear_screen() -> None:
    """Clear the console screen."""
    console.clear()


def print_banner(text: str) -> None:
    """
    Print a banner with the given text.

    Args:
        text: Banner text
    """
    print_panel(
        Text(text, justify="center", style="bold cyan"),
        style="cyan",
    )
