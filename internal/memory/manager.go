package memory

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
)

// Manager bundles the memory Reader, Writer, and Summarizer.
type Manager struct {
	reader     *Reader
	writer     *Writer
	summarizer *Summarizer
	signals    *SignalStore
	indexDir   string
	log        *zap.Logger
}

// NewManager creates a fully initialized Manager.
func NewManager(memoryRoot, apiKey string, log *zap.Logger) *Manager {
	reader := NewReader(memoryRoot, "", DefaultMinScore, log)

	indexDir := filepath.Join(memoryRoot, "index")
	if err := reader.Index().LoadFromDisk(indexDir); err != nil {
		log.Info("memory: no persisted index, will build from scratch")
	} else {
		log.Info("memory: loaded persisted index", zap.Int("chunks", reader.Index().Size()))
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := reader.EnsureIndex(ctx); err != nil {
		log.Warn("memory index build failed, memory disabled", zap.Error(err))
		return nil
	}

	if err := reader.Index().SaveToDisk(indexDir); err != nil {
		log.Warn("memory: failed to persist index", zap.Error(err))
	}

	m := &Manager{reader: reader, signals: NewSignalStore(memoryRoot), indexDir: indexDir, log: log}

	if pruned := m.signals.PruneStale(DefaultValidDurationDays); pruned > 0 {
		log.Info("memory: pruned stale signals", zap.Int("count", pruned))
	}

	writer, err := NewWriter(memoryRoot, log)
	if err != nil {
		log.Warn("memory writer init failed", zap.Error(err))
	} else {
		m.writer = writer
	}

	if apiKey != "" {
		m.summarizer = NewSummarizer(memoryRoot, apiKey, m.signals, log)
		m.summarizer.Run(context.Background())
	}

	log.Info("long-term memory enabled", zap.String("root", memoryRoot))
	return m
}

// SearchAndFormat searches memory by UUID and returns a formatted context string.
func (m *Manager) SearchAndFormat(ctx context.Context, query string, uuid string) string {
	results, err := m.reader.SearchDaily(ctx, query, 3, uuid)
	if err != nil {
		m.log.Warn("memory search failed", zap.Error(err))
		return ""
	}

	for _, r := range results {
		m.signals.Record(r.Text, r.Score, query)
	}

	return m.reader.FormatContext(results, 0, uuid)
}

// RecordInteraction writes the user message to the daily log and hot-updates the index.
func (m *Manager) RecordInteraction(ctx context.Context, voiceInput, uuid, name string) {
	if m.writer == nil {
		return
	}
	m.writer.AppendInteraction(voiceInput, uuid, name)
	if m.reader.IndexReady() {
		m.writer.AppendToIndex(ctx, m.reader.Index(), voiceInput, uuid)
	}
}

// MaybeSummarize triggers background summarization.
func (m *Manager) Summarize() {
	if m.summarizer != nil && m.summarizer.CheckEligibility() {
		go func() {
			m.summarizer.Run(context.Background())
			if err := m.reader.Index().SaveToDisk(m.indexDir); err != nil {
				m.log.Warn("memory: failed to persist index", zap.Error(err))
			}
		}()
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
