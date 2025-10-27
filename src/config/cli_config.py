"""CLI interface configuration.

This module provides CLIConfig dataclass for managing CLI display options,
behavior settings, and user interface preferences.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLIConfig:
    """Configuration for CLI interface.

    Attributes:
        colorize: Enable colored output in terminal
        show_tokens: Display token counts for inputs/outputs
        show_latency: Display generation time metrics
        show_reasoning: Display reasoning traces from analysis channel
        verbose: Enable debug logging output
        auto_save: Automatically save conversation history on exit
        history_file: Path to conversation history file
        enable_autocomplete: Enable command autocompletion
        multiline_input: Enable multi-line message input
    """

    # Display
    colorize: bool = True
    show_tokens: bool = True
    show_latency: bool = True
    show_reasoning: bool = False
    verbose: bool = False

    # Behavior
    auto_save: bool = False
    history_file: str = ".chat_history.json"

    # Advanced
    enable_autocomplete: bool = True
    multiline_input: bool = False

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate history file path
        if not self.history_file:
            raise ValueError("history_file cannot be empty")

        # Ensure history file has valid extension
        valid_extensions = {".json", ".jsonl", ".txt"}
        history_path = Path(self.history_file)
        if history_path.suffix not in valid_extensions:
            raise ValueError(
                f"history_file must have one of {valid_extensions}, got {history_path.suffix}"
            )

        # Check if history file parent directory is writable (if it exists)
        parent_dir = history_path.parent
        if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
            raise ValueError(
                f"history_file parent directory is not writable: {parent_dir}"
            )

    @classmethod
    def from_env(cls) -> "CLIConfig":
        """Load configuration from environment variables.

        Environment variables:
            COLORIZE: Enable colored output (default: true)
            SHOW_TOKENS: Display token counts (default: true)
            SHOW_LATENCY: Display generation time (default: true)
            SHOW_REASONING: Display reasoning traces (default: false)
            VERBOSE: Enable debug logging (default: false)
            AUTO_SAVE: Auto-save conversation history (default: false)
            HISTORY_FILE: Conversation history file path (default: .chat_history.json)
            ENABLE_AUTOCOMPLETE: Enable command completion (default: true)
            MULTILINE_INPUT: Enable multi-line input (default: false)

        Returns:
            CLIConfig instance with values from environment.
        """
        # Helper to parse boolean env vars
        def get_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ("true", "1", "yes", "on")

        return cls(
            colorize=get_bool("COLORIZE", True),
            show_tokens=get_bool("SHOW_TOKENS", True),
            show_latency=get_bool("SHOW_LATENCY", True),
            show_reasoning=get_bool("SHOW_REASONING", False),
            verbose=get_bool("VERBOSE", False),
            auto_save=get_bool("AUTO_SAVE", False),
            history_file=os.getenv("HISTORY_FILE", ".chat_history.json"),
            enable_autocomplete=get_bool("ENABLE_AUTOCOMPLETE", True),
            multiline_input=get_bool("MULTILINE_INPUT", False),
        )
