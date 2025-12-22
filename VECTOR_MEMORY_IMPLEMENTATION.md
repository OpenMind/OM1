# Vector Memory Implementation for OM1

This implementation addresses issue #856: Long-term Conversational Memory for Single-Agent Mode.

## Overview

Vector Memory provides persistent, semantic-searchable conversation history that survives robot restarts. It works alongside the existing `history_length` parameter to give robots both short-term working memory and long-term recall capabilities.

## Features

- **Persistent Storage**: Conversations survive robot restarts
- **Semantic Search**: Finds relevant memories by meaning, not keywords
- **Configurable**: Easy enable/disable, adjustable recall limits
- **Non-Breaking**: Completely optional, backward compatible
- **Efficient**: Uses sentence-transformers for fast embeddings

## Architecture

### Components

1. **VectorMemoryProvider** (`src/providers/vector_memory_provider.py`)
   - Manages conversation storage and retrieval
   - Uses Qdrant for vector database
   - Implements semantic search with sentence-transformers

2. **OpenAI LLM Integration** (`src/llm/plugins/openai_llm.py`)
   - Retrieves relevant memories before LLM calls
   - Stores conversations after robot responds
   - Injects memory context into prompts

### How It Works
```
User Message
    ↓
Vector Memory: Search for relevant past conversations
    ↓
LLM: Receives current message + relevant memories
    ↓
Robot Response
    ↓
Vector Memory: Store this conversation turn
```

## Configuration

Add to your robot config (e.g., `config/conversation_with_memory.json5`):
```json5
{
  "cortex_llm": {
    "type": "OpenAILLM",
    "config": {
      "agent_name": "Spot",
      "history_length": 10,

      "vector_memory": {
        "enabled": true,
        "collection_name": "spot_memories",
        "embedding_model": "all-MiniLM-L6-v2",
        "max_recall": 3,
        "storage_path": "./data/spot_vector_memory"
      }
    }
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable vector memory |
| `collection_name` | string | `"om1_conversations"` | Unique name for this robot's memories |
| `embedding_model` | string | `"all-MiniLM-L6-v2"` | Sentence-transformers model |
| `max_recall` | integer | `3` | Max memories to retrieve per query |
| `storage_path` | string | `"./data/vector_memory"` | Persistent storage location |

## Usage Example

### Day 1: User introduces themselves
```
User: "Hi, I'm Sarah. I usually need coffee around 6 AM."
Robot: "Nice to meet you, Sarah! I'll remember that."
```

### Day 2: Robot restarts, user returns
```
User: "Can you make my usual morning drink?"
Robot: [Recalls: "Sarah drinks coffee at 6 AM"]
      "Of course, Sarah! Your coffee at 6 AM as usual."
```

## Performance

- **Embedding Generation**: ~50ms per message
- **Semantic Search**: ~10ms for top-3 results
- **Storage**: Negligible overhead
- **Memory**: ~100MB for embedding model

## Testing

### Run Unit Tests
```bash
pytest tests/providers/test_vector_memory.py -v
```

## Dependencies
```
sentence-transformers>=2.2.0
qdrant-client>=1.7.0
```

## Future Enhancements

- Cloud storage backend support
- Memory importance scoring
- Automatic memory consolidation
- Multi-modal memory (images, audio)
- Privacy controls (forget functionality)

## Credits

Implementation by @Wanbogang
Issue: #856
