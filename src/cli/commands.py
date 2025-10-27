"""
Command parser and dispatcher for CLI interface.

Handles command parsing, validation, and execution for the REPL.
"""

from typing import Dict, Callable, Optional, Tuple, TYPE_CHECKING, Any
from dataclasses import dataclass
import json
from pathlib import Path

if TYPE_CHECKING:
    from ..conversation.history import ConversationManager
    from ..conversation.persistence import ConversationPersistence


@dataclass
class Command:
    """Command definition with help text and handler"""
    name: str
    description: str
    usage: str
    handler: Optional[Callable] = None


# Command definitions (Wave 2: all commands)
COMMANDS: Dict[str, Command] = {
    "/help": Command(
        name="/help",
        description="Show available commands",
        usage="/help [command]",
    ),
    "/quit": Command(
        name="/quit",
        description="Exit the program",
        usage="/quit",
    ),
    "/clear": Command(
        name="/clear",
        description="Clear conversation history",
        usage="/clear",
    ),
    "/history": Command(
        name="/history",
        description="Show last n messages (default: 10)",
        usage="/history [n]",
    ),
    "/save": Command(
        name="/save",
        description="Save conversation to file",
        usage="/save <filepath>",
    ),
    "/load": Command(
        name="/load",
        description="Load conversation from file",
        usage="/load <filepath>",
    ),
    "/export": Command(
        name="/export",
        description="Export conversation (markdown/json/text)",
        usage="/export <format> <filepath>",
    ),
    "/info": Command(
        name="/info",
        description="Show model info, stats, and memory",
        usage="/info",
    ),
    "/switch": Command(
        name="/switch",
        description="Switch to a different model",
        usage="/switch <model>",
    ),
    "/stats": Command(
        name="/stats",
        description="Show conversation statistics",
        usage="/stats",
    ),
    "/search": Command(
        name="/search",
        description="Search conversation history",
        usage="/search <query>",
    ),
}


class CommandParser:
    """Parse and validate user commands"""

    def __init__(self):
        self.commands = COMMANDS

    def is_command(self, text: str) -> bool:
        """
        Check if input is a command.

        Args:
            text: User input

        Returns:
            True if input starts with '/'
        """
        return text.strip().startswith("/")

    def parse(self, text: str) -> Tuple[str, list]:
        """
        Parse command and arguments.

        Args:
            text: Command string

        Returns:
            Tuple of (command_name, arguments)

        Examples:
            "/help" -> ("/help", [])
            "/help /quit" -> ("/help", ["/quit"])
            "/save chat.json" -> ("/save", ["chat.json"])
        """
        parts = text.strip().split()
        command = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        return command, args

    def validate(self, command: str) -> bool:
        """
        Validate if command exists.

        Args:
            command: Command name

        Returns:
            True if command is valid
        """
        return command in self.commands

    def get_help(self, command: Optional[str] = None) -> str:
        """
        Get help text for command(s).

        Args:
            command: Optional specific command, if None returns all commands

        Returns:
            Formatted help text
        """
        if command and command in self.commands:
            cmd = self.commands[command]
            return f"{cmd.name} - {cmd.description}\nUsage: {cmd.usage}"

        # Return all commands
        help_text = ["Available commands:\n"]

        for cmd_name, cmd in sorted(self.commands.items()):
            help_text.append(f"  {cmd.name:15} {cmd.description}")

        help_text.append("\nType a message to chat, or a command to execute.")

        return "\n".join(help_text)


class CommandDispatcher:
    """Execute commands and route to appropriate handlers"""

    def __init__(
        self,
        conversation: Optional['ConversationManager'] = None,
        persistence: Optional['ConversationPersistence'] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize command dispatcher.

        Args:
            conversation: ConversationManager instance for history operations
            persistence: ConversationPersistence instance for save/load operations
            model_info: Dictionary containing model information
        """
        self.parser = CommandParser()
        self.conversation = conversation
        self.persistence = persistence
        self.model_info = model_info or {}

    def execute(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Execute a command.

        Args:
            text: Command text

        Returns:
            Tuple of (continue_loop, output_message)
            - continue_loop: True to continue REPL, False to exit
            - output_message: Optional message to display
        """
        command, args = self.parser.parse(text)

        # Validate command
        if not self.parser.validate(command):
            return True, f"Unknown command: {command}\nType /help for available commands."

        # Execute command
        if command == "/help":
            help_cmd = args[0] if args else None
            return True, self.parser.get_help(help_cmd)

        elif command == "/quit":
            return False, "Goodbye!"

        elif command == "/clear":
            return self._execute_clear()

        elif command == "/history":
            n = int(args[0]) if args else 10
            return self._execute_history(n)

        elif command == "/save":
            if not args:
                return True, "Error: /save requires a filepath argument.\nUsage: /save <filepath>"
            return self._execute_save(args[0])

        elif command == "/load":
            if not args:
                return True, "Error: /load requires a filepath argument.\nUsage: /load <filepath>"
            return self._execute_load(args[0])

        elif command == "/export":
            if len(args) < 2:
                return True, "Error: /export requires format and filepath arguments.\nUsage: /export <format> <filepath>"
            return self._execute_export(args[0], args[1])

        elif command == "/info":
            return self._execute_info()

        elif command == "/switch":
            if not args:
                return True, "Error: /switch requires a model name.\nUsage: /switch <model>"
            return self._execute_switch(args[0])

        elif command == "/stats":
            return self._execute_stats()

        elif command == "/search":
            if not args:
                return True, "Error: /search requires a query.\nUsage: /search <query>"
            query = " ".join(args)
            return self._execute_search(query)

        # Default (should not reach here)
        return True, f"Command {command} not implemented yet."

    def _execute_clear(self) -> Tuple[bool, Optional[str]]:
        """Clear conversation history."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        message_count = self.conversation.get_message_count()
        self.conversation.clear()
        return True, f"Cleared {message_count} messages from conversation history."

    def _execute_history(self, n: int) -> Tuple[bool, Optional[str]]:
        """Show last n messages."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        messages = self.conversation.get_history()

        if not messages:
            return True, "No messages in conversation history."

        # Get last n messages
        last_n = messages[-n:] if n < len(messages) else messages

        # Format output
        lines = [f"Last {len(last_n)} messages:\n"]
        for i, msg in enumerate(last_n, 1):
            role_display = msg.role.upper()
            lines.append(f"[{i}] {role_display} ({msg.tokens} tokens)")
            lines.append(f"{msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            lines.append("")

        return True, "\n".join(lines)

    def _execute_save(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Save conversation to file."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        if not self.persistence:
            # Fallback to direct save
            try:
                self.conversation.save(filepath)
                return True, f"Saved conversation to {filepath}"
            except Exception as e:
                return True, f"Error saving conversation: {str(e)}"

        try:
            # Extract filename from path
            filename = Path(filepath).name
            saved_path = self.persistence.save_conversation(self.conversation, filename)
            return True, f"Saved conversation to {saved_path}"
        except Exception as e:
            return True, f"Error saving conversation: {str(e)}"

    def _execute_load(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """Load conversation from file."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        if not self.persistence:
            # Fallback to direct load
            try:
                from ..conversation.history import ConversationManager
                loaded = ConversationManager.load(filepath)
                # Replace current conversation
                self.conversation.messages = loaded.messages
                self.conversation.max_context_tokens = loaded.max_context_tokens
                self.conversation.created_at = loaded.created_at
                self.conversation.metadata = loaded.metadata
                return True, f"Loaded {loaded.get_message_count()} messages from {filepath}"
            except Exception as e:
                return True, f"Error loading conversation: {str(e)}"

        try:
            filename = Path(filepath).name
            loaded = self.persistence.load_conversation(filename)
            # Replace current conversation
            self.conversation.messages = loaded.messages
            self.conversation.max_context_tokens = loaded.max_context_tokens
            self.conversation.created_at = loaded.created_at
            self.conversation.metadata = loaded.metadata
            return True, f"Loaded {loaded.get_message_count()} messages from {filepath}"
        except Exception as e:
            return True, f"Error loading conversation: {str(e)}"

    def _execute_export(self, format_type: str, filepath: str) -> Tuple[bool, Optional[str]]:
        """Export conversation in specified format."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        format_lower = format_type.lower()

        try:
            if format_lower == "json":
                # Export as JSON
                data = self.conversation.to_dict()
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True, f"Exported conversation to {filepath} (JSON format)"

            elif format_lower == "text":
                # Export as plain text
                if not self.persistence:
                    return True, "Error: Persistence manager required for text export."
                self.persistence.export_to_text(self.conversation, filepath)
                return True, f"Exported conversation to {filepath} (text format)"

            elif format_lower == "markdown" or format_lower == "md":
                # Export as markdown
                if not self.persistence:
                    return True, "Error: Persistence manager required for markdown export."
                self.persistence.export_to_markdown(self.conversation, filepath)
                return True, f"Exported conversation to {filepath} (markdown format)"

            else:
                return True, f"Error: Unsupported format '{format_type}'. Use: json, text, or markdown"

        except Exception as e:
            return True, f"Error exporting conversation: {str(e)}"

    def _execute_info(self) -> Tuple[bool, Optional[str]]:
        """Show model and system information."""
        lines = ["System Information:\n"]

        # Model info
        if self.model_info:
            lines.append("Model:")
            for key, value in self.model_info.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        # Conversation info
        if self.conversation:
            lines.append("Conversation:")
            lines.append(f"  Messages: {self.conversation.get_message_count()}")
            lines.append(f"  Total tokens: {self.conversation.get_token_count()}")
            lines.append(f"  Max context: {self.conversation.max_context_tokens}")
            lines.append(f"  Created: {self.conversation.created_at}")

        return True, "\n".join(lines)

    def _execute_switch(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """Switch to a different model."""
        # This is a placeholder - actual model switching will be implemented
        # by model-agent in their ModelSwitcher class
        return True, f"Model switching not yet implemented. Requested model: {model_name}"

    def _execute_stats(self) -> Tuple[bool, Optional[str]]:
        """Show conversation statistics."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        messages = self.conversation.get_history()

        if not messages:
            return True, "No conversation statistics available (empty conversation)."

        # Calculate statistics
        user_msgs = [m for m in messages if m.role == "user"]
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        system_msgs = [m for m in messages if m.role == "system"]

        total_chars = sum(len(m.content) for m in messages)
        avg_msg_length = total_chars / len(messages) if messages else 0

        lines = [
            "Conversation Statistics:\n",
            f"Total messages: {len(messages)}",
            f"  User: {len(user_msgs)}",
            f"  Assistant: {len(assistant_msgs)}",
            f"  System: {len(system_msgs)}",
            "",
            f"Total tokens: {self.conversation.get_token_count()}",
            f"Total characters: {total_chars:,}",
            f"Average message length: {avg_msg_length:.1f} chars",
            "",
            f"Context usage: {self.conversation.get_token_count()}/{self.conversation.max_context_tokens} tokens "
            f"({100 * self.conversation.get_token_count() / self.conversation.max_context_tokens:.1f}%)",
        ]

        return True, "\n".join(lines)

    def _execute_search(self, query: str) -> Tuple[bool, Optional[str]]:
        """Search conversation history."""
        if not self.conversation:
            return True, "Error: No conversation manager available."

        messages = self.conversation.get_history()

        if not messages:
            return True, "No messages to search."

        # Simple case-insensitive search
        query_lower = query.lower()
        results = []

        for i, msg in enumerate(messages, 1):
            if query_lower in msg.content.lower():
                results.append((i, msg))

        if not results:
            return True, f"No results found for query: '{query}'"

        # Format results
        lines = [f"Found {len(results)} results for '{query}':\n"]

        for msg_num, msg in results:
            role_display = msg.role.upper()
            # Show context around match
            content = msg.content
            idx = content.lower().index(query_lower)
            start = max(0, idx - 40)
            end = min(len(content), idx + len(query) + 40)
            excerpt = content[start:end]
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(content):
                excerpt = excerpt + "..."

            lines.append(f"[{msg_num}] {role_display}:")
            lines.append(f"  {excerpt}")
            lines.append("")

        return True, "\n".join(lines)


def format_command_help() -> str:
    """
    Format help text for all commands.

    Returns:
        Formatted help string
    """
    parser = CommandParser()
    return parser.get_help()
