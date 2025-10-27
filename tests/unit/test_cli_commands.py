"""
Unit tests for CLI command system.

Tests all Wave 2 commands including:
- Command parsing and validation
- Command execution
- Error handling
- Integration with conversation and persistence
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.cli.commands import (
    Command,
    CommandParser,
    CommandDispatcher,
    COMMANDS,
)
from src.conversation.history import ConversationManager
from src.conversation.persistence import ConversationPersistence


class TestCommand:
    """Test Command dataclass"""

    def test_command_creation(self):
        """Test creating a Command"""
        cmd = Command(
            name="/test",
            description="Test command",
            usage="/test [arg]",
        )
        assert cmd.name == "/test"
        assert cmd.description == "Test command"
        assert cmd.usage == "/test [arg]"
        assert cmd.handler is None


class TestCommandParser:
    """Test CommandParser"""

    def test_is_command(self):
        """Test command detection"""
        parser = CommandParser()
        assert parser.is_command("/help")
        assert parser.is_command("/quit")
        assert not parser.is_command("hello")
        assert not parser.is_command("")

    def test_parse_simple_command(self):
        """Test parsing command without arguments"""
        parser = CommandParser()
        cmd, args = parser.parse("/help")
        assert cmd == "/help"
        assert args == []

    def test_parse_command_with_args(self):
        """Test parsing command with arguments"""
        parser = CommandParser()
        cmd, args = parser.parse("/save my_file.json")
        assert cmd == "/save"
        assert args == ["my_file.json"]

        cmd, args = parser.parse("/export markdown output.md")
        assert cmd == "/export"
        assert args == ["markdown", "output.md"]

    def test_parse_empty_string(self):
        """Test parsing empty string"""
        parser = CommandParser()
        cmd, args = parser.parse("")
        assert cmd == ""
        assert args == []

    def test_validate_command(self):
        """Test command validation"""
        parser = CommandParser()
        assert parser.validate("/help")
        assert parser.validate("/quit")
        assert parser.validate("/clear")
        assert parser.validate("/stats")
        assert not parser.validate("/invalid")
        assert not parser.validate("not_a_command")

    def test_get_help_all_commands(self):
        """Test getting help for all commands"""
        parser = CommandParser()
        help_text = parser.get_help()
        assert "Available commands" in help_text
        assert "/help" in help_text
        assert "/quit" in help_text
        assert "/clear" in help_text
        assert "/stats" in help_text

    def test_get_help_specific_command(self):
        """Test getting help for specific command"""
        parser = CommandParser()
        help_text = parser.get_help("/help")
        assert "/help" in help_text
        assert "Show available commands" in help_text
        assert "/help [command]" in help_text

    def test_all_wave2_commands_registered(self):
        """Test that all Wave 2 commands are registered"""
        expected_commands = [
            "/help", "/quit", "/clear", "/history",
            "/save", "/load", "/export", "/info",
            "/switch", "/stats", "/search"
        ]
        parser = CommandParser()
        for cmd in expected_commands:
            assert cmd in parser.commands


class TestCommandDispatcher:
    """Test CommandDispatcher"""

    @pytest.fixture
    def conversation(self):
        """Create test conversation"""
        conv = ConversationManager(max_context_tokens=4096)
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        conv.add_message("user", "How are you?")
        conv.add_message("assistant", "I'm doing well, thanks!")
        return conv

    @pytest.fixture
    def dispatcher(self, conversation):
        """Create test dispatcher"""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ConversationPersistence(storage_dir=tmpdir)
            model_info = {
                "model_name": "test-model",
                "context_length": 4096,
            }
            yield CommandDispatcher(
                conversation=conversation,
                persistence=persistence,
                model_info=model_info,
            )

    def test_execute_help(self, dispatcher):
        """Test /help command"""
        continue_loop, output = dispatcher.execute("/help")
        assert continue_loop is True
        assert "Available commands" in output
        assert "/help" in output

    def test_execute_help_specific(self, dispatcher):
        """Test /help with specific command"""
        continue_loop, output = dispatcher.execute("/help /quit")
        assert continue_loop is True
        assert "/quit" in output
        assert "Exit the program" in output

    def test_execute_quit(self, dispatcher):
        """Test /quit command"""
        continue_loop, output = dispatcher.execute("/quit")
        assert continue_loop is False
        assert "Goodbye" in output

    def test_execute_clear(self, dispatcher, conversation):
        """Test /clear command"""
        initial_count = conversation.get_message_count()
        assert initial_count > 0

        continue_loop, output = dispatcher.execute("/clear")
        assert continue_loop is True
        assert "Cleared" in output
        assert conversation.get_message_count() == 0

    def test_execute_stats(self, dispatcher, conversation):
        """Test /stats command"""
        continue_loop, output = dispatcher.execute("/stats")
        assert continue_loop is True
        assert "Conversation Statistics" in output
        assert "Total messages" in output
        assert "Total tokens" in output

    def test_execute_info(self, dispatcher):
        """Test /info command"""
        continue_loop, output = dispatcher.execute("/info")
        assert continue_loop is True
        assert "System Information" in output
        assert "test-model" in output

    def test_execute_history(self, dispatcher):
        """Test /history command"""
        continue_loop, output = dispatcher.execute("/history 2")
        assert continue_loop is True
        assert "Last" in output
        assert "messages" in output

    def test_execute_history_default(self, dispatcher):
        """Test /history with default count"""
        continue_loop, output = dispatcher.execute("/history")
        assert continue_loop is True
        assert "messages" in output

    def test_execute_search(self, dispatcher):
        """Test /search command"""
        continue_loop, output = dispatcher.execute("/search Hello")
        assert continue_loop is True
        assert "Found" in output or "No results" in output

    def test_execute_search_no_results(self, dispatcher):
        """Test /search with no matches"""
        continue_loop, output = dispatcher.execute("/search NONEXISTENT_TEXT_12345")
        assert continue_loop is True
        assert "No results found" in output

    def test_execute_save(self, dispatcher):
        """Test /save command"""
        filename = "test_save.json"

        continue_loop, output = dispatcher.execute(f"/save {filename}")
        assert continue_loop is True
        assert "Saved" in output

        # Verify file was created in persistence storage directory
        saved_path = Path(dispatcher.persistence.storage_dir) / filename
        assert saved_path.exists()

        # Verify content
        with open(saved_path, 'r') as f:
            data = json.load(f)
            assert "messages" in data
            assert "message_count" in data

    def test_execute_save_no_filepath(self, dispatcher):
        """Test /save without filepath argument"""
        continue_loop, output = dispatcher.execute("/save")
        assert continue_loop is True
        assert "Error" in output
        assert "filepath" in output.lower()

    def test_execute_load(self, dispatcher, conversation):
        """Test /load command"""
        filename = "test_load.json"

        # Save first using persistence
        saved_path = dispatcher.persistence.save_conversation(conversation, filename)

        # Clear conversation
        conversation.clear()
        assert conversation.get_message_count() == 0

        # Load
        continue_loop, output = dispatcher.execute(f"/load {filename}")
        assert continue_loop is True
        assert "Loaded" in output
        assert conversation.get_message_count() > 0

    def test_execute_load_no_filepath(self, dispatcher):
        """Test /load without filepath argument"""
        continue_loop, output = dispatcher.execute("/load")
        assert continue_loop is True
        assert "Error" in output
        assert "filepath" in output.lower()

    def test_execute_export_json(self, dispatcher):
        """Test /export command with JSON format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            continue_loop, output = dispatcher.execute(f"/export json {filepath}")
            assert continue_loop is True
            assert "Exported" in output

            # Verify file
            assert Path(filepath).exists()
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert "messages" in data
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_execute_export_markdown(self, dispatcher):
        """Test /export command with markdown format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            filepath = f.name

        try:
            continue_loop, output = dispatcher.execute(f"/export markdown {filepath}")
            assert continue_loop is True
            assert "Exported" in output

            # Verify file
            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_execute_export_text(self, dispatcher):
        """Test /export command with text format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filepath = f.name

        try:
            continue_loop, output = dispatcher.execute(f"/export text {filepath}")
            assert continue_loop is True
            assert "Exported" in output

            # Verify file
            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_execute_export_invalid_format(self, dispatcher):
        """Test /export with invalid format"""
        continue_loop, output = dispatcher.execute("/export invalid output.txt")
        assert continue_loop is True
        assert "Unsupported format" in output

    def test_execute_export_no_args(self, dispatcher):
        """Test /export without arguments"""
        continue_loop, output = dispatcher.execute("/export")
        assert continue_loop is True
        assert "Error" in output

    def test_execute_switch(self, dispatcher):
        """Test /switch command (placeholder)"""
        continue_loop, output = dispatcher.execute("/switch phi-3-mini")
        assert continue_loop is True
        assert "not yet implemented" in output or "phi-3-mini" in output

    def test_execute_switch_no_model(self, dispatcher):
        """Test /switch without model name"""
        continue_loop, output = dispatcher.execute("/switch")
        assert continue_loop is True
        assert "Error" in output

    def test_execute_invalid_command(self, dispatcher):
        """Test invalid command"""
        continue_loop, output = dispatcher.execute("/invalid")
        assert continue_loop is True
        assert "Unknown command" in output

    def test_dispatcher_without_conversation(self):
        """Test dispatcher without conversation manager"""
        dispatcher = CommandDispatcher()

        # Commands that require conversation should fail gracefully
        continue_loop, output = dispatcher.execute("/stats")
        assert continue_loop is True
        assert "Error" in output

        continue_loop, output = dispatcher.execute("/clear")
        assert continue_loop is True
        assert "Error" in output

    def test_dispatcher_without_persistence(self, conversation):
        """Test dispatcher without persistence manager"""
        dispatcher = CommandDispatcher(conversation=conversation)

        # Save/load should still work with fallback
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            continue_loop, output = dispatcher.execute(f"/save {filepath}")
            assert continue_loop is True
            # Should work even without persistence manager
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestCommandIntegration:
    """Integration tests for command system"""

    def test_full_workflow(self):
        """Test complete command workflow"""
        # Create conversation
        conv = ConversationManager()
        conv.add_message("user", "Test message 1")
        conv.add_message("assistant", "Response 1")

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ConversationPersistence(storage_dir=tmpdir)
            dispatcher = CommandDispatcher(
                conversation=conv,
                persistence=persistence,
            )

            # Get stats
            _, output = dispatcher.execute("/stats")
            assert "2" in output  # 2 messages

            # Save conversation
            _, output = dispatcher.execute("/save test.json")
            assert "Saved" in output

            # Clear
            _, output = dispatcher.execute("/clear")
            assert conv.get_message_count() == 0

            # Load
            _, output = dispatcher.execute("/load test.json")
            assert conv.get_message_count() == 2

            # Export
            _, output = dispatcher.execute("/export json test_export.json")
            assert "Exported" in output

    def test_error_handling(self):
        """Test error handling in commands"""
        dispatcher = CommandDispatcher()

        # Commands that need conversation should fail gracefully
        commands_needing_conversation = [
            "/clear", "/stats", "/history", "/search test",
            "/save test.json", "/load test.json", "/export json test.json"
        ]

        for cmd in commands_needing_conversation:
            continue_loop, output = dispatcher.execute(cmd)
            assert continue_loop is True
            assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
