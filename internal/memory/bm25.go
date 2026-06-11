package memory

import (
	"math"
	"strings"
	"sync"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// BM25Index is a lightweight in-memory BM25 ranking index.
type BM25Index struct {
	mu       sync.RWMutex
	docs     map[string]bm25Doc
	inverted map[string][]bm25Posting
	avgDL    float64
}

type bm25Doc struct {
	entry  MemoryEntry
	length int
}

type bm25Posting struct {
	hash string
	tf   int
}

// NewBM25Index creates an empty BM25 index.
func NewBM25Index() *BM25Index {
	return &BM25Index{
		docs:     make(map[string]bm25Doc),
		inverted: make(map[string][]bm25Posting),
	}
}

// Add inserts a document into the index. No-op if hash already exists.
func (idx *BM25Index) Add(hash string, entry MemoryEntry) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.docs[hash]; exists {
		return
	}

	tokens := tokenize(entry.Text)
	idx.docs[hash] = bm25Doc{entry: entry, length: len(tokens)}

	// Count term frequencies.
	tf := make(map[string]int)
	for _, t := range tokens {
		tf[t]++
	}

	for term, count := range tf {
		idx.inverted[term] = append(idx.inverted[term], bm25Posting{hash: hash, tf: count})
	}

	// Recompute average document length.
	total := 0
	for _, d := range idx.docs {
		total += d.length
	}
	idx.avgDL = float64(total) / float64(len(idx.docs))
}

// Search returns the top-k documents ranked by BM25 score.
func (idx *BM25Index) Search(query string, topK int, userID string) []MemoryEntry {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	n := float64(len(idx.docs))
	if n == 0 {
		return nil
	}

	queryTerms := tokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}

	scores := make(map[string]float64)
	for _, term := range queryTerms {
		postings, ok := idx.inverted[term]
		if !ok {
			continue
		}
		df := float64(len(postings))
		idf := math.Log((n-df+0.5)/(df+0.5) + 1)

		for _, p := range postings {
			doc := idx.docs[p.hash]
			tf := float64(p.tf)
			dl := float64(doc.length)
			score := idf * (tf * (bm25K1 + 1)) / (tf + bm25K1*(1-bm25B+bm25B*dl/idx.avgDL))
			scores[p.hash] += score
		}
	}

	if len(scores) == 0 {
		return nil
	}

	// Sort by score descending.
	type scored struct {
		hash  string
		score float64
	}
	var sorted []scored
	for h, s := range scores {
		sorted = append(sorted, scored{h, s})
	}
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j].score > sorted[j-1].score; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	if len(sorted) > topK {
		sorted = sorted[:topK]
	}

	results := make([]MemoryEntry, len(sorted))
	for i, s := range sorted {
		e := idx.docs[s.hash].entry
		e.Score = s.score
		results[i] = e
	}
	return results
}

// Size returns the number of indexed documents.
func (idx *BM25Index) Size() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

func tokenize(text string) []string {
	return strings.Fields(strings.ToLower(text))
}
