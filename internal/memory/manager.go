package memory

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
)

// Manager bundles the memory Reader, Writer, and Summarizer into
// a single facade used by the runtime.
type Manager struct {
	reader     *Reader
	writer     *Writer
	summarizer *Summarizer
	log        *zap.Logger
}

// NewManager creates a fully initialized Manager. Returns nil if the
// index build fails (matching the KB pattern of warn-and-continue).
func NewManager(memoryRoot, apiKey string, log *zap.Logger) *Manager {
	reader := NewReader(memoryRoot, "", DefaultMinScore, log)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := reader.EnsureIndex(ctx); err != nil {
		log.Warn("memory index build failed, memory disabled", zap.Error(err))
		return nil
	}

	m := &Manager{reader: reader, log: log}

	writer, err := NewWriter(memoryRoot, log)
	if err != nil {
		log.Warn("memory writer init failed", zap.Error(err))
	} else {
		m.writer = writer
	}

	if apiKey != "" {
		m.summarizer = NewSummarizer(memoryRoot, apiKey, log)
		m.summarizer.Run(context.Background())
	}

	log.Info("long-term memory enabled", zap.String("root", memoryRoot))
	return m
}

// SearchAndFormat resolves the current user, searches memory, and returns
// a formatted context string ready for prompt injection.
func (m *Manager) SearchAndFormat(ctx context.Context, query string, userID string) string {
	results, err := m.reader.SearchDaily(ctx, query, 3, userID)
	if err != nil {
		m.log.Warn("memory search failed", zap.Error(err))
		return ""
	}
	return m.reader.FormatContext(results, 0, userID)
}

// RecordInteraction writes the user message to the daily log and hot-updates
// the index. No-op if the writer was not initialized.
func (m *Manager) RecordInteraction(ctx context.Context, voiceInput string, userID string) {
	if m.writer == nil {
		return
	}
	m.writer.AppendInteraction(voiceInput, userID)
	if m.reader.IndexReady() {
		m.writer.AppendToIndex(ctx, m.reader.Index(), voiceInput, userID)
	}
}

// MaybeSummarize triggers background summarization if enough new
// interactions have accumulated. No-op if the summarizer is nil.
func (m *Manager) Summarize() {
	if m.summarizer != nil && m.summarizer.CheckEligibility() {
		go m.summarizer.Run(context.Background())
	}
}

// ResolveMemoryRoot locates the memory directory relative to cwd.
func ResolveMemoryRoot() string {
	if cwd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(cwd, "memory")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
		return candidate
	}
	return "memory"
}
