# OM1 Conversation Exporter

Export agent conversations to Markdown format for debugging, documentation, and analysis.

## Features

- Export conversations to Markdown format
- Human-readable output
- Easy to share and document
- Timestamps and structured format

## Usage

### Basic Export
```bash
python export_conversation.py --example --format markdown
```

This creates a `conversations/` directory with the exported conversation.

### Example Output
```
conversations/
  └── spot_20260130_121500.md
```

## Output Format

The exported Markdown file includes:
- Session information (agent name, session ID, timestamp)
- Each exchange with:
  - User input
  - Vision output
  - LLM decision
  - Action result

## Requirements

- Python 3.8+
- No additional dependencies required

## Future Enhancements

- Support for JSON, CSV, HTML formats
- Real-time conversation logging
- Session filtering and search
- Integration with OM1 logging system

## Contributing

This is a prototype implementation. Feedback and suggestions welcome!