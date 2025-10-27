"""
Enhanced input handling with prompt_toolkit.

Provides rich input features including auto-completion, history,
and multi-line editing.
"""

from typing import Optional, List
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.history import InMemoryHistory, FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.document import Document
from pathlib import Path


# Define prompt style
prompt_style = Style.from_dict({
    'prompt': '#00aaff bold',
    'user-input': '#ffffff',
    'completion-menu.completion': 'bg:#008888 #ffffff',
    'completion-menu.completion.current': 'bg:#00aaaa #000000',
    'scrollbar.background': 'bg:#88aaaa',
    'scrollbar.button': 'bg:#222222',
})


class CommandCompleter(Completer):
    """Smart command completer with contextual suggestions"""

    def __init__(self, commands: List[str]):
        """
        Initialize command completer.

        Args:
            commands: List of command names
        """
        self.commands = commands

    def get_completions(self, document: Document, complete_event):
        """
        Get command completions based on current input.

        Args:
            document: Current document
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor
        words = text.split()

        # Only complete if text starts with /
        if not text.startswith('/'):
            return

        # Get the command being typed
        if len(words) == 0:
            current_word = ""
        else:
            current_word = words[0]

        # Find matching commands
        for cmd in self.commands:
            if cmd.startswith(current_word):
                # Calculate how many characters to complete
                start_position = -len(current_word) if current_word else 0
                yield Completion(
                    cmd,
                    start_position=start_position,
                    display=cmd,
                    display_meta=self._get_command_description(cmd)
                )

    def _get_command_description(self, cmd: str) -> str:
        """Get brief description for command"""
        descriptions = {
            "/help": "Show commands",
            "/quit": "Exit program",
            "/clear": "Clear history",
            "/history": "Show messages",
            "/save": "Save conversation",
            "/load": "Load conversation",
            "/export": "Export conversation",
            "/info": "System info",
            "/switch": "Change model",
            "/stats": "Show statistics",
            "/search": "Search history",
        }
        return descriptions.get(cmd, "")


class EnhancedInput:
    """Enhanced input with prompt_toolkit features"""

    def __init__(
        self,
        commands: Optional[List[str]] = None,
        history_file: Optional[str] = None,
        multiline: bool = False,
    ):
        """
        Initialize enhanced input.

        Args:
            commands: Optional list of commands for auto-completion
            history_file: Optional file to persist input history
            multiline: Enable multi-line input mode
        """
        # Setup history
        if history_file:
            history_path = Path(history_file)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history = FileHistory(str(history_path))
        else:
            self.history = InMemoryHistory()

        # Setup command auto-completion if commands provided
        self.completer = None
        if commands:
            self.completer = CommandCompleter(commands)

        # Setup key bindings for multi-line
        self.multiline = multiline
        self.kb = self._create_key_bindings()

        # Create session
        self.session = PromptSession(
            history=self.history,
            style=prompt_style,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Create custom key bindings"""
        kb = KeyBindings()

        # Meta+Enter or Esc+Enter for multi-line submit
        @kb.add('escape', 'enter')
        @kb.add('c-j')  # Ctrl+J as alternative
        def _(event):
            """Submit multi-line input"""
            event.current_buffer.validate_and_handle()

        return kb

    def get_input(
        self,
        prompt: str = "> ",
        multiline: Optional[bool] = None,
    ) -> str:
        """
        Get user input with enhanced features.

        Args:
            prompt: Prompt string to display
            multiline: Override default multiline setting

        Returns:
            User input string

        Raises:
            KeyboardInterrupt: If user presses Ctrl+C
            EOFError: If user presses Ctrl+D
        """
        use_multiline = multiline if multiline is not None else self.multiline

        try:
            # Add help text for multiline mode
            if use_multiline:
                full_prompt = f"{prompt}(Esc+Enter to send, Ctrl+C to cancel)\n> "
            else:
                full_prompt = prompt

            user_input = self.session.prompt(
                full_prompt,
                completer=self.completer,
                multiline=use_multiline,
                key_bindings=self.kb if use_multiline else None,
                complete_while_typing=True,
                enable_history_search=True,
            )
            return user_input.strip()

        except KeyboardInterrupt:
            # Re-raise to allow graceful shutdown
            raise

        except EOFError:
            # Ctrl+D pressed - treat as quit
            raise

    def toggle_multiline(self) -> None:
        """Toggle multi-line input mode"""
        self.multiline = not self.multiline

    def add_to_history(self, text: str) -> None:
        """
        Manually add text to history.

        Args:
            text: Text to add to history
        """
        self.history.append_string(text)


class SimpleInput:
    """Fallback simple input (no prompt_toolkit)"""

    def __init__(self, commands: Optional[List[str]] = None):
        """
        Initialize simple input.

        Args:
            commands: Unused (for API compatibility)
        """
        pass

    def get_input(self, prompt: str = "> ") -> str:
        """
        Get user input using built-in input().

        Args:
            prompt: Prompt string to display

        Returns:
            User input string

        Raises:
            KeyboardInterrupt: If user presses Ctrl+C
            EOFError: If user presses Ctrl+D
        """
        return input(prompt).strip()

    def add_to_history(self, text: str) -> None:
        """
        No-op for simple input.

        Args:
            text: Unused
        """
        pass


def create_input_handler(
    commands: Optional[List[str]] = None,
    use_enhanced: bool = True,
    history_file: Optional[str] = None,
    multiline: bool = False,
) -> 'EnhancedInput | SimpleInput':
    """
    Create appropriate input handler.

    Args:
        commands: Optional list of commands for auto-completion
        use_enhanced: Whether to use enhanced input (prompt_toolkit)
        history_file: Optional file to persist input history
        multiline: Enable multi-line input mode

    Returns:
        Input handler instance
    """
    if use_enhanced:
        try:
            return EnhancedInput(
                commands=commands,
                history_file=history_file,
                multiline=multiline,
            )
        except ImportError:
            # Fall back to simple input if prompt_toolkit not available
            return SimpleInput(commands=commands)
    else:
        return SimpleInput(commands=commands)
