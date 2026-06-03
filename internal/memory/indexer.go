package memory

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/knowledgebase"
)

const (
	DefaultMinScore          = 0.3
	DefaultValidDurationDays = 14
)

// MemoryIndex is an in-memory embedding index for memory chunks.
type MemoryIndex struct {
	mu       sync.RWMutex
	cache    map[string]indexEntry
	embedder knowledgebase.Embedder
	log      *zap.Logger
}

type indexEntry struct {
	embedding []float32
	doc       MemoryEntry
}

// NewMemoryIndex creates a new index backed by the given embedder.
func NewMemoryIndex(embedder knowledgebase.Embedder, log *zap.Logger) *MemoryIndex {
	return &MemoryIndex{
		cache:    make(map[string]indexEntry),
		embedder: embedder,
		log:      log,
	}
}

// Size returns the number of cached chunks.
func (idx *MemoryIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.cache)
}

// Search finds the top-k most similar chunks to query using cosine similarity.
func (idx *MemoryIndex) Search(ctx context.Context, query string, topK int, minScore float64, userID string) ([]MemoryEntry, error) {
	idx.mu.RLock()
	cacheSize := len(idx.cache)
	idx.mu.RUnlock()

	if cacheSize == 0 || strings.TrimSpace(query) == "" {
		return nil, nil
	}

	queryVec, err := idx.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("memory index: embed query: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	type scored struct {
		score float64
		doc   MemoryEntry
	}
	var results []scored

	for _, entry := range idx.cache {
		if userID != "" && entry.doc.Metadata["user_id"] != userID {
			continue
		}
		s := cosineSimilarity(queryVec, entry.embedding)
		if s >= minScore {
			results = append(results, scored{score: s, doc: entry.doc})
		}
	}

	// Sort descending by score.
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].score > results[j-1].score; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}

	if len(results) > topK {
		results = results[:topK]
	}

	docs := make([]MemoryEntry, len(results))
	for i, r := range results {
		d := r.doc
		d.Score = r.score
		docs[i] = d
	}

	idx.log.Info("memory search",
		zap.String("query", truncate(query, 50)),
		zap.Int("results", len(docs)),
		zap.Int("total_chunks", cacheSize),
	)
	return docs, nil
}

// LoadChunksBatch loads multiple chunks into the index in one batch on startup.
func (idx *MemoryIndex) LoadChunksBatch(ctx context.Context, chunks []MemoryEntry) (int, error) {
	// Identify new chunks.
	idx.mu.RLock()
	var newChunks []MemoryEntry
	var newHashes []string
	for _, c := range chunks {
		h := hashText(c.Text)
		if _, exists := idx.cache[h]; !exists {
			newChunks = append(newChunks, c)
			newHashes = append(newHashes, h)
		}
	}
	idx.mu.RUnlock()

	if len(newChunks) == 0 {
		return 0, nil
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	loaded := 0
	for i, chunk := range newChunks {
		vec, err := idx.embedder.Embed(ctx, chunk.Text)
		if err != nil {
			idx.log.Warn("memory index: batch embed failed, partial load",
				zap.Int("loaded", loaded), zap.Error(err))
			return loaded, err
		}
		idx.cache[newHashes[i]] = indexEntry{embedding: vec, doc: chunk}
		loaded++
	}

	idx.log.Info("memory index: loaded chunks",
		zap.Int("new", loaded),
		zap.Int("total", len(idx.cache)),
	)
	return loaded, nil
}

// AddChunk embeds and inserts a single chunk (hot update). Returns true if added.
func (idx *MemoryIndex) AddChunk(ctx context.Context, chunk MemoryEntry) (bool, error) {
	h := hashText(chunk.Text)

	idx.mu.RLock()
	_, exists := idx.cache[h]
	idx.mu.RUnlock()
	if exists {
		return false, nil
	}

	vec, err := idx.embedder.Embed(ctx, chunk.Text)
	if err != nil {
		return false, fmt.Errorf("memory index: hot update: %w", err)
	}

	idx.mu.Lock()
	idx.cache[h] = indexEntry{embedding: vec, doc: chunk}
	idx.mu.Unlock()

	idx.log.Debug("memory index: hot updated", zap.Int("total", idx.Size()))
	return true, nil
}

// ParseDailyFile parses a daily markdown file into MemoryEntry chunks.
// Each ## section becomes a separate chunk.
func ParseDailyFile(path string) ([]MemoryEntry, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("memory: read %s: %w", path, err)
	}

	base := filepath.Base(path)
	dateStem := strings.TrimSuffix(base, filepath.Ext(base))
	datePrefix := "[Date: " + dateStem + "]"

	userTagRe := regexp.MustCompile(`^\[User: (.+)\]$`)

	var chunks []MemoryEntry
	var currentChunk strings.Builder
	var currentUserID string

	lines := strings.Split(string(content), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "## ") && currentChunk.Len() > 0 {
			text := strings.TrimSpace(currentChunk.String())
			if text != "" {
				meta := map[string]string{"source": base}
				if currentUserID != "" {
					meta["user_id"] = currentUserID
				}
				chunks = append(chunks, MemoryEntry{
					Text:     datePrefix + "\n" + text,
					Metadata: meta,
				})
			}
			currentChunk.Reset()
			currentUserID = ""
		}

		if m := userTagRe.FindStringSubmatch(line); m != nil {
			currentUserID = strings.ToLower(strings.TrimSpace(m[1]))
		}
		currentChunk.WriteString(line)
		currentChunk.WriteByte('\n')
	}

	// Flush last chunk.
	if text := strings.TrimSpace(currentChunk.String()); text != "" {
		meta := map[string]string{"source": base}
		if currentUserID != "" {
			meta["user_id"] = currentUserID
		}
		chunks = append(chunks, MemoryEntry{
			Text:     datePrefix + "\n" + text,
			Metadata: meta,
		})
	}

	return chunks, nil
}

// BuildIndex populates the given MemoryIndex from recent daily markdown files.
// Files older than validDays are deleted.
func BuildIndex(ctx context.Context, idx *MemoryIndex, dailyDir string, validDays int) error {
	entries, err := os.ReadDir(dailyDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	cutoff := time.Now().AddDate(0, 0, -validDays)
	var allChunks []MemoryEntry

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".md") {
			continue
		}
		stem := strings.TrimSuffix(entry.Name(), ".md")
		fileDate, parseErr := time.Parse("2006-01-02", stem)
		if parseErr != nil {
			continue
		}

		filePath := filepath.Join(dailyDir, entry.Name())
		if fileDate.Before(cutoff) {
			_ = os.Remove(filePath)
			idx.log.Info("memory: deleted expired daily log", zap.String("file", entry.Name()))
			continue
		}

		chunks, parseErr := ParseDailyFile(filePath)
		if parseErr != nil {
			idx.log.Warn("memory: failed to parse daily file", zap.String("file", entry.Name()), zap.Error(parseErr))
			continue
		}
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) > 0 {
		loaded, loadErr := idx.LoadChunksBatch(ctx, allChunks)
		if loadErr != nil {
			return loadErr
		}
		idx.log.Info("memory: populated index",
			zap.Int("chunks", len(allChunks)),
			zap.Int("new", loaded),
			zap.Int("valid_days", validDays),
		)
	}
	return nil
}

// --- helpers ---

func hashText(text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", h)
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (normA * normB)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
