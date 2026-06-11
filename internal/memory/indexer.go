package memory

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/coder/hnsw"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/knowledgebase"
)

const (
	DefaultMinScore          = 0.5
	DefaultValidDurationDays = 60
)

// MemoryIndex is an in-memory HNSW index for memory chunks.
type MemoryIndex struct {
	mu       sync.RWMutex
	graph    *hnsw.Graph[int]
	docs     map[int]MemoryEntry
	hashes   map[string]int
	nextID   int
	bm25     *BM25Index
	embedder knowledgebase.Embedder
	log      *zap.Logger
}

// NewMemoryIndex creates a new index backed by the given embedder.
func NewMemoryIndex(embedder knowledgebase.Embedder, log *zap.Logger) *MemoryIndex {
	return &MemoryIndex{
		graph:    hnsw.NewGraph[int](),
		docs:     make(map[int]MemoryEntry),
		hashes:   make(map[string]int),
		bm25:     NewBM25Index(),
		embedder: embedder,
		log:      log,
	}
}

// Size returns the number of indexed chunks.
func (idx *MemoryIndex) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

// Search finds the top-k most similar chunks to query using HNSW ANN search.
func (idx *MemoryIndex) Search(ctx context.Context, query string, topK int, minScore float64, userID string) ([]MemoryEntry, error) {
	idx.mu.RLock()
	totalChunks := len(idx.docs)
	idx.mu.RUnlock()

	if totalChunks == 0 || strings.TrimSpace(query) == "" {
		return nil, nil
	}

	queryVec, err := idx.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("memory index: embed query: %w", err)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	searchK := topK * 3
	if searchK > len(idx.docs) {
		searchK = len(idx.docs)
	}

	nodes := idx.graph.Search(queryVec, searchK)

	var results []MemoryEntry
	for _, node := range nodes {
		doc, ok := idx.docs[node.Key]
		if !ok {
			continue
		}
		similarity := 1 - float64(idx.graph.Distance(queryVec, node.Value))
		if similarity >= minScore {
			doc.Score = similarity
			results = append(results, doc)
		}
		if len(results) >= topK {
			break
		}
	}

	idx.log.Info("memory search",
		zap.String("query", truncate(query, 50)),
		zap.Int("results", len(results)),
		zap.Int("total_chunks", totalChunks),
	)
	return results, nil
}

const (
	fusionAlpha = 0.7 // embedding weight
	fusionBeta  = 0.3 // BM25 weight
	userBoost   = 0.1 // soft boost for matching user_id
)

// HybridSearch combines embedding cosine similarity with BM25 lexical scoring.
func (idx *MemoryIndex) HybridSearch(ctx context.Context, query string, topK int, minScore float64, userID string) ([]MemoryEntry, error) {
	embeddingResults, err := idx.Search(ctx, query, topK*2, minScore, userID)
	if err != nil {
		return nil, err
	}

	bm25Results := idx.bm25.Search(query, topK*2, userID)

	// Build score maps keyed by text hash.
	type fusedEntry struct {
		doc            MemoryEntry
		embeddingScore float64
		bm25Score      float64
	}
	merged := make(map[string]*fusedEntry)

	for _, r := range embeddingResults {
		h := hashText(r.Text)
		merged[h] = &fusedEntry{doc: r, embeddingScore: r.Score}
	}
	for _, r := range bm25Results {
		h := hashText(r.Text)
		if e, ok := merged[h]; ok {
			e.bm25Score = r.Score
		} else {
			merged[h] = &fusedEntry{doc: r, bm25Score: r.Score}
		}
	}

	if len(merged) == 0 {
		return nil, nil
	}

	// Min-max normalize each signal.
	var eMin, eMax, bMin, bMax float64
	first := true
	for _, e := range merged {
		if first {
			eMin, eMax = e.embeddingScore, e.embeddingScore
			bMin, bMax = e.bm25Score, e.bm25Score
			first = false
		}
		if e.embeddingScore < eMin {
			eMin = e.embeddingScore
		}
		if e.embeddingScore > eMax {
			eMax = e.embeddingScore
		}
		if e.bm25Score < bMin {
			bMin = e.bm25Score
		}
		if e.bm25Score > bMax {
			bMax = e.bm25Score
		}
	}

	normalize := func(v, min, max float64) float64 {
		if max == min {
			return 0
		}
		return (v - min) / (max - min)
	}

	type scored struct {
		doc   MemoryEntry
		score float64
	}
	var results []scored
	for _, e := range merged {
		final := fusionAlpha*normalize(e.embeddingScore, eMin, eMax) +
			fusionBeta*normalize(e.bm25Score, bMin, bMax)

		if userID != "" && e.doc.Metadata["user_id"] == userID {
			final += userBoost
		}

		d := e.doc
		d.Score = final
		results = append(results, scored{doc: d, score: final})
	}

	// Sort descending.
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
		docs[i] = r.doc
	}

	idx.log.Info("memory hybrid search",
		zap.String("query", truncate(query, 50)),
		zap.Int("embedding_hits", len(embeddingResults)),
		zap.Int("bm25_hits", len(bm25Results)),
		zap.Int("fused", len(docs)),
	)
	return docs, nil
}

// LoadChunksBatch loads multiple chunks into the index in one batch on startup.
func (idx *MemoryIndex) LoadChunksBatch(ctx context.Context, chunks []MemoryEntry) (int, error) {
	idx.mu.RLock()
	var newChunks []MemoryEntry
	var newHashes []string
	for _, c := range chunks {
		h := hashText(c.Text)
		if _, exists := idx.hashes[h]; !exists {
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
		id := idx.nextID
		idx.nextID++
		idx.graph.Add(hnsw.MakeNode(id, vec))
		idx.docs[id] = chunk
		idx.hashes[newHashes[i]] = id
		idx.bm25.Add(newHashes[i], chunk)
		loaded++
	}

	idx.log.Info("memory index: loaded chunks",
		zap.Int("new", loaded),
		zap.Int("total", len(idx.docs)),
	)
	return loaded, nil
}

// AddChunk embeds and inserts a single chunk (hot update). Returns true if added.
func (idx *MemoryIndex) AddChunk(ctx context.Context, chunk MemoryEntry) (bool, error) {
	h := hashText(chunk.Text)

	idx.mu.RLock()
	_, exists := idx.hashes[h]
	idx.mu.RUnlock()
	if exists {
		return false, nil
	}

	vec, err := idx.embedder.Embed(ctx, chunk.Text)
	if err != nil {
		return false, fmt.Errorf("memory index: hot update: %w", err)
	}

	idx.mu.Lock()
	id := idx.nextID
	idx.nextID++
	idx.graph.Add(hnsw.MakeNode(id, vec))
	idx.docs[id] = chunk
	idx.hashes[h] = id
	idx.mu.Unlock()

	idx.bm25.Add(h, chunk)

	idx.log.Debug("memory index: hot updated", zap.Int("total", idx.Size()))
	return true, nil
}

type persistedEntry struct {
	ID       int               `json:"id"`
	Hash     string            `json:"hash"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata"`
}

type persistedIndex struct {
	NextID  int              `json:"next_id"`
	Entries []persistedEntry `json:"entries"`
}

// SaveToDisk persists the HNSW graph and document metadata to dir.
func (idx *MemoryIndex) SaveToDisk(dir string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("memory index: create dir: %w", err)
	}

	// Save HNSW graph.
	graphPath := filepath.Join(dir, "index.graph")
	f, err := os.Create(graphPath)
	if err != nil {
		return fmt.Errorf("memory index: create graph file: %w", err)
	}
	defer func() { _ = f.Close() }()
	w := bufio.NewWriter(f)
	if err := idx.graph.Export(w); err != nil {
		return fmt.Errorf("memory index: export graph: %w", err)
	}
	if err := w.Flush(); err != nil {
		return fmt.Errorf("memory index: flush graph: %w", err)
	}

	// Save document metadata.
	idToHash := make(map[int]string, len(idx.hashes))
	for h, id := range idx.hashes {
		idToHash[id] = h
	}
	entries := make([]persistedEntry, 0, len(idx.docs))
	for id, doc := range idx.docs {
		entries = append(entries, persistedEntry{
			ID: id, Hash: idToHash[id], Text: doc.Text, Metadata: doc.Metadata,
		})
	}
	data, err := json.Marshal(persistedIndex{NextID: idx.nextID, Entries: entries})
	if err != nil {
		return fmt.Errorf("memory index: marshal metadata: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "index.meta.json"), data, 0o644); err != nil {
		return fmt.Errorf("memory index: write metadata: %w", err)
	}

	idx.log.Info("memory index: saved to disk", zap.Int("chunks", len(idx.docs)))
	return nil
}

// LoadFromDisk restores the HNSW graph and document metadata from dir.
func (idx *MemoryIndex) LoadFromDisk(dir string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Load HNSW graph.
	graphPath := filepath.Join(dir, "index.graph")
	f, err := os.Open(graphPath)
	if err != nil {
		return err
	}
	graph := hnsw.NewGraph[int]()
	if err := graph.Import(bufio.NewReader(f)); err != nil {
		_ = f.Close()
		return fmt.Errorf("memory index: import graph: %w", err)
	}
	_ = f.Close()

	// Load document metadata.
	raw, err := os.ReadFile(filepath.Join(dir, "index.meta.json"))
	if err != nil {
		return fmt.Errorf("memory index: read metadata: %w", err)
	}
	var meta persistedIndex
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fmt.Errorf("memory index: parse metadata: %w", err)
	}

	idx.graph = graph
	idx.nextID = meta.NextID
	for _, e := range meta.Entries {
		doc := MemoryEntry{Text: e.Text, Metadata: e.Metadata}
		idx.docs[e.ID] = doc
		idx.hashes[e.Hash] = e.ID
		idx.bm25.Add(e.Hash, doc)
	}

	idx.log.Info("memory index: loaded from disk", zap.Int("chunks", len(idx.docs)))
	return nil
}

// PruneHashes removes entries whose hash is not in validHashes.
func (idx *MemoryIndex) PruneHashes(validHashes map[string]struct{}) int {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	pruned := 0
	for h, id := range idx.hashes {
		if _, ok := validHashes[h]; !ok {
			delete(idx.docs, id)
			delete(idx.hashes, h)
			pruned++
		}
	}
	return pruned
}

// ParseDailyFile parses a daily markdown file into MemoryEntry chunks.
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
			idx.log.Info("deleted expired daily log", zap.String("file", entry.Name()))
			continue
		}

		chunks, parseErr := ParseDailyFile(filePath)
		if parseErr != nil {
			idx.log.Warn("failed to parse daily file", zap.String("file", entry.Name()), zap.Error(parseErr))
			continue
		}
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) > 0 || idx.Size() > 0 {
		validHashes := make(map[string]struct{}, len(allChunks))
		for _, c := range allChunks {
			validHashes[hashText(c.Text)] = struct{}{}
		}
		if pruned := idx.PruneHashes(validHashes); pruned > 0 {
			idx.log.Info("pruned stale entries", zap.Int("count", pruned))
		}
	}

	if len(allChunks) > 0 {
		loaded, loadErr := idx.LoadChunksBatch(ctx, allChunks)
		if loadErr != nil {
			return loadErr
		}
		idx.log.Info("populated index",
			zap.Int("chunks", len(allChunks)),
			zap.Int("new", loaded),
			zap.Int("valid_days", validDays),
		)
	}
	return nil
}

func hashText(text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", h)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
