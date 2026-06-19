// Package memoryeval is a thin public façade over OM1's internal long-term
// memory packages, so external tools (e.g. the LoCoMo benchmark harness) can
// drive the memory system without importing OM1's internal/ packages — which
// Go forbids across module boundaries.
//
// The memory implementation itself stays in internal/memory; this package only
// re-exports the constructors, types, and constants a benchmark driver needs.
package memoryeval

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/knowledgebase"
	"github.com/openmind/om1/internal/memory"
)

// Re-exported types. These are aliases, so values flow transparently between
// this package and internal/memory and all exported methods remain callable.
type (
	// Reader searches and reads long-term memory.
	Reader = memory.Reader
	// MemoryIndex is the HNSW + BM25 hybrid index.
	MemoryIndex = memory.MemoryIndex
	// Summarizer runs the LLM fact-extraction pipeline.
	Summarizer = memory.Summarizer
	// MemoryEntry is a single retrievable chunk with its score.
	MemoryEntry = memory.MemoryEntry
)

const (
	// DefaultMinScore is the default minimum similarity for embedding hits.
	DefaultMinScore = memory.DefaultMinScore
	// DefaultEmbedderBaseURL is the default embedding-service base URL.
	DefaultEmbedderBaseURL = knowledgebase.DefaultBaseURL
)

// NewIndex builds an empty in-memory index backed by the HTTP embedder at
// embedderURL (empty string uses DefaultEmbedderBaseURL).
func NewIndex(embedderURL string, log *zap.Logger) *MemoryIndex {
	return memory.NewMemoryIndex(knowledgebase.NewHTTPEmbedder(embedderURL), log)
}

// BuildIndex populates idx from daily markdown files under dailyDir, keeping
// files newer than validDays.
func BuildIndex(ctx context.Context, idx *MemoryIndex, dailyDir string, validDays int) error {
	return memory.BuildIndex(ctx, idx, dailyDir, validDays)
}

// NewReader creates a memory Reader. Call SetIndex (or EnsureIndex) before searching.
func NewReader(memoryRoot, embedderURL string, minScore float64, log *zap.Logger) *Reader {
	return memory.NewReader(memoryRoot, embedderURL, minScore, log)
}

// NewSummarizer creates a Summarizer with a fresh signal store rooted at memoryRoot.
func NewSummarizer(memoryRoot, apiKey string, log *zap.Logger) *Summarizer {
	return memory.NewSummarizer(memoryRoot, apiKey, memory.NewSignalStore(memoryRoot), log)
}
