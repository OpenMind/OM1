package memory

import (
	"context"

	"go.uber.org/zap"

	mem "github.com/openmind/om1/internal/memory"
)

type (
	Reader     = mem.Reader
	Index      = mem.MemoryIndex
	Summarizer = mem.Summarizer
	Entry      = mem.MemoryEntry
)

const (
	DefaultMinScore        = mem.DefaultMinScore
	DefaultEmbedderBaseURL = mem.DefaultEmbedderBaseURL
)

func NewIndex(embedderURL string, log *zap.Logger) *Index {
	return mem.NewIndexFromURL(embedderURL, log)
}

func BuildIndex(ctx context.Context, idx *Index, dailyDir string, validDays int) error {
	return mem.BuildIndex(ctx, idx, dailyDir, validDays)
}

func NewReader(memoryRoot, embedderURL string, minScore float64, log *zap.Logger) *Reader {
	return mem.NewReader(memoryRoot, embedderURL, minScore, log)
}

func NewSummarizer(memoryRoot, apiKey string, log *zap.Logger) *Summarizer {
	return mem.NewSummarizerForRoot(memoryRoot, apiKey, log)
}
