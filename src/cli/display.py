"""
Message formatting and display utilities.

Handles formatting of chat messages, metadata display,
and other UI elements.
"""

from typing import Dict, Optional, List, Iterator
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text
from .output import (
    console,
    print_user_message,
    print_assistant_message,
    print_system_message,
    print_separator,
    print_panel,
)


class MessageFormatter:
    """Format and display chat messages"""

    def __init__(self, show_metadata: bool = True):
        """
        Initialize message formatter.

        Args:
            show_metadata: Whether to show message metadata
        """
        self.show_metadata = show_metadata

    def display_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Display formatted user message.

        Args:
            content: Message content
            metadata: Optional message metadata
        """
        print_user_message(content)

        if self.show_metadata and metadata:
            self._display_metadata(metadata)

    def display_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        streaming: bool = False,
    ) -> None:
        """
        Display formatted assistant message.

        Args:
            content: Message content
            metadata: Optional message metadata
            streaming: Whether this is a streaming response
        """
        if streaming:
            # For streaming, print without newline (will be handled by caller)
            console.print("[green]Assistant:[/green] ", end="")
            console.print(content, end="")
        else:
            print_assistant_message(content)

        if self.show_metadata and metadata and not streaming:
            self._display_metadata(metadata)

    def display_reasoning(self, reasoning: str, max_length: int = 500) -> None:
        """Display reasoning trace from analysis channel.

        Args:
            reasoning: Raw reasoning text from analysis channel
            max_length: Maximum length before truncation
        """
        from src.prompts.harmony_channels import format_reasoning_trace

        if not reasoning:
            return

        # Format and truncate reasoning
        formatted = format_reasoning_trace(reasoning, max_length)

        # Display with distinct styling
        console.print()
        console.print("─" * 50, style="dim cyan")
        console.print("🧠 Reasoning:", style="bold cyan")
        console.print(formatted, style="dim italic cyan")
        console.print("─" * 50, style="dim cyan")

    def display_system_message(self, content: str) -> None:
        """
        Display formatted system message.

        Args:
            content: Message content
        """
        print_system_message(content)

    def _display_metadata(self, metadata: Dict) -> None:
        """
        Display message metadata.

        Args:
            metadata: Metadata dictionary
        """
        meta_parts = []

        if "model" in metadata:
            meta_parts.append(f"model: {metadata['model']}")

        if "latency_ms" in metadata:
            meta_parts.append(f"latency: {metadata['latency_ms']}ms")

        if "tokens" in metadata:
            tokens = metadata["tokens"]
            if isinstance(tokens, dict):
                total = tokens.get("total_tokens", 0)
                meta_parts.append(f"tokens: {total}")
            else:
                meta_parts.append(f"tokens: {tokens}")

        if "provider" in metadata:
            meta_parts.append(f"provider: {metadata['provider']}")

        if meta_parts:
            meta_text = " | ".join(meta_parts)
            console.print(f"[dim]({meta_text})[/dim]")


class ConversationDisplay:
    """Display full conversation history"""

    def __init__(self):
        """Initialize conversation display"""
        self.formatter = MessageFormatter(show_metadata=False)

    def display_history(self, messages: List[Dict[str, str]]) -> None:
        """
        Display conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
        """
        print_separator()
        print_system_message("Conversation History:")
        print_separator()

        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                self.formatter.display_user_message(content)
            elif role == "assistant":
                self.formatter.display_assistant_message(content)
            elif role == "system":
                self.formatter.display_system_message(content)

            # Add separator between messages (except last)
            if i < len(messages):
                print_separator(char="·")

        print_separator()

    def display_summary(self, messages: List[Dict[str, str]]) -> None:
        """
        Display conversation summary.

        Args:
            messages: List of message dicts
        """
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        total_chars = sum(len(m.get("content", "")) for m in messages)

        summary = f"""
**Conversation Summary**

- User messages: {user_count}
- Assistant messages: {assistant_count}
- Total characters: {total_chars:,}
        """

        print_panel(summary.strip(), title="Summary", style="blue")


class WelcomeDisplay:
    """Display welcome banner and initial messages"""

    @staticmethod
    def display_welcome(model_name: Optional[str] = None) -> None:
        """
        Display welcome banner.

        Args:
            model_name: Optional model name to display
        """
        banner_text = "GPT-OSS CLI Chat"

        if model_name:
            banner_text += f"\nModel: {model_name}"

        print_panel(
            banner_text,
            title="Welcome",
            style="cyan",
        )

        print_system_message("Type your message to start chatting, or /help for commands.")
        print_separator()

    @staticmethod
    def display_loading(message: str = "Loading model...") -> None:
        """
        Display loading message.

        Args:
            message: Loading message to display
        """
        console.print(f"[yellow]{message}[/yellow]")

    @staticmethod
    def display_ready() -> None:
        """Display ready message"""
        print_system_message("✓ Ready to chat!")
        print_separator()


class StatisticsDisplay:
    """Display statistics and metrics"""

    @staticmethod
    def display_token_usage(usage: Dict[str, int]) -> None:
        """
        Display token usage statistics.

        Args:
            usage: Token usage dict with prompt_tokens, completion_tokens, total_tokens
        """
        stats = f"""
**Token Usage**

- Prompt tokens: {usage.get('prompt_tokens', 0):,}
- Completion tokens: {usage.get('completion_tokens', 0):,}
- Total tokens: {usage.get('total_tokens', 0):,}
        """

        print_panel(stats.strip(), title="Statistics", style="blue")

    @staticmethod
    def display_performance_metrics(metrics: Dict) -> None:
        """
        Display performance metrics.

        Args:
            metrics: Performance metrics dict
        """
        lines = ["**Performance Metrics**\n"]

        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            elif isinstance(value, int):
                lines.append(f"- {key}: {value:,}")
            else:
                lines.append(f"- {key}: {value}")

        print_panel("\n".join(lines), title="Metrics", style="blue")

    @staticmethod
    def display_generation_metrics(
        tokens: int,
        duration_ms: float,
        tokens_per_second: Optional[float] = None
    ) -> None:
        """
        Display generation performance metrics.

        Args:
            tokens: Number of tokens generated
            duration_ms: Generation duration in milliseconds
            tokens_per_second: Optional pre-calculated tokens/sec
        """
        if tokens_per_second is None and duration_ms > 0:
            tokens_per_second = (tokens / duration_ms) * 1000

        metrics_text = f"[dim]({tokens} tokens, {duration_ms:.0f}ms"
        if tokens_per_second:
            metrics_text += f", {tokens_per_second:.1f} tok/s"
        metrics_text += ")[/dim]"

        console.print(metrics_text)


class ProgressDisplay:
    """Display progress indicators for long-running operations"""

    @staticmethod
    def loading_spinner(description: str = "Loading...") -> Progress:
        """
        Create a loading spinner.

        Args:
            description: Description text to display

        Returns:
            Progress instance to use as context manager
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )

    @staticmethod
    def progress_bar(description: str = "Processing...") -> Progress:
        """
        Create a progress bar.

        Args:
            description: Description text to display

        Returns:
            Progress instance to use as context manager
        """
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )


class StreamingDisplay:
    """Display streaming text generation with live updates"""

    def __init__(self, show_role: bool = True):
        """
        Initialize streaming display.

        Args:
            show_role: Whether to show role label
        """
        self.show_role = show_role
        self.token_count = 0
        self.start_time: Optional[float] = None

    def stream_tokens(
        self,
        token_stream: Iterator[str],
        role: str = "assistant"
    ) -> str:
        """
        Display streaming tokens in real-time.

        Args:
            token_stream: Iterator yielding token strings
            role: Message role for display

        Returns:
            Complete generated text
        """
        import time

        # Show role label
        if self.show_role:
            role_color = "green" if role == "assistant" else "cyan"
            console.print(f"[{role_color}]{role.capitalize()}:[/{role_color}] ", end="")

        # Stream tokens
        full_text = ""
        self.token_count = 0
        self.start_time = time.time()

        try:
            for token in token_stream:
                console.print(token, end="")
                full_text += token
                self.token_count += 1

            console.print()  # Newline after streaming

            # Show metrics
            duration_ms = (time.time() - self.start_time) * 1000
            tokens_per_sec = (self.token_count / duration_ms) * 1000 if duration_ms > 0 else 0
            StatisticsDisplay.display_generation_metrics(
                self.token_count,
                duration_ms,
                tokens_per_sec
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Generation interrupted[/yellow]")
            raise

        return full_text

    def stream_with_live_metrics(
        self,
        token_stream: Iterator[str],
        role: str = "assistant"
    ) -> str:
        """
        Display streaming tokens with live metrics updates.

        Args:
            token_stream: Iterator yielding token strings
            role: Message role for display

        Returns:
            Complete generated text
        """
        import time

        full_text = ""
        self.token_count = 0
        self.start_time = time.time()

        # Create display text
        text = Text()
        role_color = "green" if role == "assistant" else "cyan"
        text.append(f"{role.capitalize()}: ", style=role_color)

        with Live(text, console=console, refresh_per_second=10) as live:
            try:
                for token in token_stream:
                    full_text += token
                    self.token_count += 1

                    # Update display
                    text = Text()
                    text.append(f"{role.capitalize()}: ", style=role_color)
                    text.append(full_text)

                    # Add metrics
                    duration = time.time() - self.start_time
                    tokens_per_sec = self.token_count / duration if duration > 0 else 0
                    text.append(f"\n[dim]({self.token_count} tokens, {tokens_per_sec:.1f} tok/s)[/dim]")

                    live.update(text)

            except KeyboardInterrupt:
                console.print("\n[yellow]Generation interrupted[/yellow]")
                raise

        # Final newline
        console.print()

        return full_text
