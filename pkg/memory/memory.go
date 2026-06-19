// Package memory is the public interface to OM1's long-term memory system.
//
// The implementation lives in internal/memory; this package re-exports it so
// external tools (e.g. the LoCoMo benchmark harness) can use the memory system
// without importing internal/ directly, which Go forbids across module
// boundaries. It carries no behavior of its own — only type aliases, constant
// aliases, and one-line forwarders to constructors in internal/memory.
package memory

import (
	"context"

	"go.uber.org/zap"

	mem "github.com/openmind/om1/internal/memory"
)

// Re-exported types. Aliases, so values flow transparently between this package
// and internal/memory and all exported methods remain callable.
type (
	// Reader searches and reads long-term memory.
	Reader = mem.Reader
	// Index is the HNSW + BM25 hybrid index.
	Index = mem.MemoryIndex
	// Summarizer runs the LLM fact-extraction pipeline.
	Summarizer = mem.Summarizer
	// Entry is a single retrievable chunk with its score.
	Entry = mem.MemoryEntry
)

const (
	// DefaultMinScore is the default minimum similarity for embedding hits.
	DefaultMinScore = mem.DefaultMinScore
	// DefaultEmbedderBaseURL is the default embedding-service base URL.
	DefaultEmbedderBaseURL = mem.DefaultEmbedderBaseURL
)

// NewIndex builds an empty in-memory index backed by the HTTP embedder at
// embedderURL (empty string uses DefaultEmbedderBaseURL).
func NewIndex(embedderURL string, log *zap.Logger) *Index {
	return mem.NewIndexFromURL(embedderURL, log)
}

// BuildIndex populates idx from daily markdown files under dailyDir, keeping
// files newer than validDays.
func BuildIndex(ctx context.Context, idx *Index, dailyDir string, validDays int) error {
	return mem.BuildIndex(ctx, idx, dailyDir, validDays)
}

// NewReader creates a memory Reader. Call SetIndex (or EnsureIndex) before searching.
func NewReader(memoryRoot, embedderURL string, minScore float64, log *zap.Logger) *Reader {
	return mem.NewReader(memoryRoot, embedderURL, minScore, log)
}

// NewSummarizer creates a Summarizer with its own signal store rooted at memoryRoot.
func NewSummarizer(memoryRoot, apiKey string, log *zap.Logger) *Summarizer {
	return mem.NewSummarizerForRoot(memoryRoot, apiKey, log)
}
