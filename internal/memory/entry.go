package memory

import "context"

// MemoryEntry holds a single memory chunk with metadata and relevance score.
type MemoryEntry struct {
	Text     string
	Metadata map[string]string
	Score    float64
}

// MemoryManager is the unified interface for both local and cloud memory.
type MemoryManager interface {
	SearchAndFormat(ctx context.Context, query, uuid string) string
	RecordInteraction(ctx context.Context, voiceInput, robotReply, uuid, name string)
	Summarize(ctx context.Context)
}
