# GPT-OSS CLI Chat - Examples

This directory contains example scripts demonstrating the conversation management features.

## Wave 2 Examples

### Export Demo (`export_demo.py`)

Demonstrates conversation export functionality:

```bash
.venv/bin/python examples/export_demo.py
```

**Features demonstrated:**
- Export to Markdown with formatted output and emojis
- Export to JSON with comprehensive statistics
- Export to plain text for human reading
- Auto-detection of format from file extension
- Configurable metadata and statistics inclusion
- Compact vs. pretty-printed JSON

**Output:** Creates files in `examples/exported_conversations/`

---

### Statistics Demo (`stats_demo.py`)

Shows conversation statistics tracking and analytics:

```bash
.venv/bin/python examples/stats_demo.py
```

**Features demonstrated:**
- Computing comprehensive conversation statistics
- Message and token breakdown by role
- Role and token percentage calculations
- Token efficiency metrics (tokens per character)
- Average response time analysis
- Token usage trends over time
- Conversation comparison
- Human-readable statistics summaries

---

### Search Demo (`search_demo.py`)

Demonstrates search and filtering capabilities:

```bash
.venv/bin/python examples/search_demo.py
```

**Features demonstrated:**
- Full-text search (case-sensitive and case-insensitive)
- Role-based filtering
- Message retrieval by ID
- Regex pattern matching for complex queries
- Filtering by token count
- Custom predicate filtering
- Context retrieval (messages with surrounding context)
- Result limiting

---

## Exported Conversations

The `exported_conversations/` directory contains example outputs in various formats:

- **`sample_conversation.md`** - Full markdown export with metadata and statistics
- **`sample_conversation.json`** - Structured JSON with comprehensive data
- **`sample_conversation.txt`** - Plain text format for easy reading
- **`no_metadata.md`** - Minimal markdown without metadata header
- **`compact.json`** - Compact JSON format (no pretty-printing)
- **`auto_detected.md`** - Example of auto-format detection

---

## Running Examples

All examples use the Python virtual environment:

```bash
# Activate venv (optional)
source .venv/bin/activate

# Run any example
python examples/export_demo.py
python examples/stats_demo.py
python examples/search_demo.py
```

Or run directly with the venv Python:

```bash
.venv/bin/python examples/export_demo.py
```

---

## Integration with CLI

These features will be integrated into the CLI REPL with commands like:

```
/export [format] [filename]  - Export current conversation
/stats                       - Show conversation statistics
/search [query]             - Search conversation history
/filter [role]              - Filter messages by role
```

(CLI integration coming in Wave 2+ for cli-agent)

---

## Example Code Snippets

### Export a Conversation

```python
from src.conversation import ConversationManager, export_to_markdown

conv = ConversationManager()
conv.add_message("user", "Hello!")
conv.add_message("assistant", "Hi there!")

# Export to markdown
export_to_markdown(conv, "my_chat.md", include_stats=True)
```

### Get Statistics

```python
from src.conversation import ConversationManager, ConversationStats

conv = ConversationManager()
# ... add messages ...

stats = ConversationStats.from_conversation(conv)
print(stats.to_summary())
print(f"User: {stats.get_role_percentage('user'):.1f}%")
```

### Search Messages

```python
from src.conversation import search_messages, filter_by_role

# Text search
results = search_messages(conv, "Python", search_role="user")

# Filter by role
user_msgs = filter_by_role(conv, "user")
```

---

## Notes

- All examples are self-contained and create their own sample data
- Exported files are written to `examples/exported_conversations/`
- Examples demonstrate real-world usage patterns
- Code follows PEP 8 and includes comprehensive docstrings

---

## Wave 1 Examples

Wave 1 examples (if any) are stored separately and demonstrate:
- Basic conversation management
- Message history
- Prompt formatting
- Token counting

See the main project README for complete documentation.
