"""
Read-Eval-Print Loop (REPL) for GPT-OSS CLI Chat.

Main interactive loop that handles user input, command processing,
and message generation.
"""

from typing import Optional, TYPE_CHECKING
import asyncio
import sys

from .commands import CommandDispatcher
from .input import create_input_handler
from .display import MessageFormatter, WelcomeDisplay
from .output import (
    print_error,
    print_system_message,
    print_separator,
    console,
)

if TYPE_CHECKING:
    from typing import Any


class REPL:
    """Read-Eval-Print Loop for chat interface"""

    def __init__(
        self,
        conversation: Optional['Any'] = None,
        engine: Optional['Any'] = None,
        config: Optional['Any'] = None,
    ):
        """
        Initialize REPL with conversation, engine, and config.

        Args:
            conversation: ConversationManager instance (optional for Wave 1)
            engine: InferenceEngine instance (optional for Wave 1)
            config: CLIConfig instance (optional for Wave 1)
        """
        self.conversation = conversation
        self.engine = engine
        self.config = config

        # Setup persistence if conversation is available
        self.persistence = None
        if conversation:
            try:
                from ..conversation.persistence import ConversationPersistence
                self.persistence = ConversationPersistence()
            except ImportError:
                pass

        # Initialize components with dependencies
        model_info = {}
        if config and hasattr(config, 'model_name'):
            model_info['model_name'] = config.model_name

        self.dispatcher = CommandDispatcher(
            conversation=conversation,
            persistence=self.persistence,
            model_info=model_info,
        )
        self.formatter = MessageFormatter(show_metadata=True)
        self.welcome = WelcomeDisplay()

        # Setup input handler with command completion and history
        command_list = list(self.dispatcher.parser.commands.keys())
        self.input_handler = create_input_handler(
            commands=command_list,
            use_enhanced=True,
            history_file=".chat_history/.repl_history",
            multiline=False,
        )

        # State
        self.running = False

    def run(self) -> None:
        """
        Start interactive REPL loop.

        Displays welcome message and enters main loop.
        Handles KeyboardInterrupt (Ctrl+C) and EOFError (Ctrl+D) gracefully.
        """
        # Display welcome banner
        model_name = None
        if self.config and hasattr(self.config, 'model_name'):
            model_name = self.config.model_name

        self.welcome.display_welcome(model_name=model_name)

        # Start loop
        self.running = True

        try:
            self._loop()
        except KeyboardInterrupt:
            print()  # New line after ^C
            print_system_message("Interrupted. Goodbye!")
        except EOFError:
            print()  # New line after ^D
            print_system_message("Goodbye!")
        finally:
            self.running = False

    def _loop(self) -> None:
        """
        Main REPL loop.

        Continuously:
        1. Gets user input
        2. Processes commands or messages
        3. Displays responses
        """
        while self.running:
            try:
                # Get user input
                user_input = self.input_handler.get_input("> ")

                # Skip empty input
                if not user_input:
                    continue

                # Process input
                should_continue = self.process_command(user_input)

                # Check if we should exit
                if not should_continue:
                    self.running = False

                # Add separator for readability
                if self.running:
                    print_separator(char="·")

            except KeyboardInterrupt:
                # Re-raise to be caught by run()
                raise
            except EOFError:
                # Re-raise to be caught by run()
                raise
            except Exception as e:
                # Handle unexpected errors
                print_error(f"An error occurred: {str(e)}")
                if self.config and hasattr(self.config, 'debug') and self.config.debug:
                    import traceback
                    traceback.print_exc()

    def process_command(self, user_input: str) -> bool:
        """
        Process user input (message or command).

        Args:
            user_input: Raw user input string

        Returns:
            True to continue loop, False to exit
        """
        # Check if input is a command
        if self.dispatcher.parser.is_command(user_input):
            # Execute command
            continue_loop, output = self.dispatcher.execute(user_input)

            # Display output if any
            if output:
                print_system_message(output)

            return continue_loop

        else:
            # Process as chat message
            return self._process_message(user_input)

    def _process_message(self, message: str) -> bool:
        """
        Process chat message.

        Args:
            message: User's chat message

        Returns:
            True to continue loop
        """
        # Display user message
        self.formatter.display_user_message(message)

        # Check if we have engine and conversation
        if not self.engine or not self.conversation:
            # Mock response for Wave 1 (no real inference yet)
            self._mock_response(message)
            return True

        # Real inference (Wave 2)
        try:
            # Add user message to conversation
            self.conversation.add_message("user", message)

            # Generate response
            response = self._generate_response()

            # Add assistant message to conversation
            self.conversation.add_message("assistant", response)

            # Display response
            self.formatter.display_assistant_message(response)

        except Exception as e:
            print_error(f"Failed to generate response: {str(e)}")

        return True

    def _generate_response(self) -> str:
        """
        Generate response using inference engine.

        Returns:
            Generated response text
        """
        # Get conversation context and format as prompt
        # ConversationManager has a format_prompt() method that converts messages to text
        prompt = self.conversation.format_prompt()

        # Generate using the formatted prompt
        response = self.engine.generate(prompt)

        return response

    def _mock_response(self, message: str) -> None:
        """
        Display mock response (for Wave 1 testing).

        Args:
            message: User message
        """
        mock_responses = {
            "hello": "Hello! I'm a GPT-OSS chatbot. How can I help you today?",
            "hi": "Hi there! What would you like to talk about?",
            "how are you": "I'm doing well, thank you for asking! I'm ready to help you with any questions.",
            "test": "Test successful! The CLI is working correctly.",
        }

        # Check for exact match
        message_lower = message.lower().strip()
        if message_lower in mock_responses:
            response = mock_responses[message_lower]
        else:
            # Default response
            response = f"You said: '{message}'. (This is a mock response - real inference coming in Wave 2!)"

        # Display mock response
        metadata = {
            "model": "mock-model",
            "latency_ms": 10,
            "tokens": {"total_tokens": 25},
            "provider": "mock",
        }

        self.formatter.display_assistant_message(response, metadata=metadata)


async def run_async_repl(
    conversation: Optional['Any'] = None,
    engine: Optional['Any'] = None,
    config: Optional['Any'] = None,
) -> None:
    """
    Async wrapper for REPL.

    Args:
        conversation: ConversationManager instance
        engine: InferenceEngine instance
        config: CLIConfig instance
    """
    repl = REPL(conversation=conversation, engine=engine, config=config)
    repl.run()


def main():
    """
    Main entry point for standalone REPL.

    For testing the REPL without full integration.
    """
    print_system_message("Starting GPT-OSS REPL (standalone mode)")
    print_system_message("Note: Full inference not available yet (Wave 1)")
    print_separator()

    repl = REPL()
    repl.run()


if __name__ == "__main__":
    main()
