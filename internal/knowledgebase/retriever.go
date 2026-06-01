package knowledgebase

import (
	"fmt"
	"os"

	faiss "github.com/DataIntelligenceCrew/go-faiss"
)

type Document struct {
	Text     string
	Metadata map[string]any
	Score    float64
}

type Retriever struct {
	index     faiss.Index
	documents []Document
	dimension int
}

// NewRetriever loads a FAISS index and associated pickle metadata from disk.
func NewRetriever(indexPath, metadataPath string) (*Retriever, error) {
	if _, err := os.Stat(indexPath); err != nil {
		return nil, fmt.Errorf("index not found: %s", indexPath)
	}
	if _, err := os.Stat(metadataPath); err != nil {
		return nil, fmt.Errorf("metadata not found: %s", metadataPath)
	}

	index, err := faiss.ReadIndex(indexPath, 0)
	if err != nil {
		return nil, fmt.Errorf("read faiss index: %w", err)
	}

	docs, err := loadPickleMetadata(metadataPath)
	if err != nil {
		index.Delete()
		return nil, fmt.Errorf("load metadata: %w", err)
	}

	dim := index.D()
	ntotal := index.Ntotal()

	if int64(len(docs)) != ntotal {
		fmt.Printf("WARNING: %d docs != %d vectors\n", len(docs), ntotal)
	}

	return &Retriever{
		index:     index,
		documents: docs,
		dimension: dim,
	}, nil
}

// Search returns the top-k most similar documents, deduplicated by answer text.
func (r *Retriever) Search(queryEmbedding []float32, topK int) ([]Document, error) {
	if len(queryEmbedding) != r.dimension {
		return nil, fmt.Errorf("query dim=%d != index dim=%d", len(queryEmbedding), r.dimension)
	}

	searchK := topK * 5
	if searchK > len(r.documents) {
		searchK = len(r.documents)
	}

	distances, labels, err := r.index.Search(queryEmbedding, int64(searchK))
	if err != nil {
		return nil, fmt.Errorf("faiss search: %w", err)
	}

	var results []Document
	seen := make(map[string]bool)

	for i, idx := range labels {
		if idx < 0 || int(idx) >= len(r.documents) {
			continue
		}
		doc := r.documents[idx]

		// Deduplicate by answer text.
		answerText := doc.Text
		if ans, ok := doc.Metadata["answer"]; ok {
			if s, ok := ans.(string); ok {
				answerText = s
			}
		}
		if seen[answerText] {
			continue
		}
		seen[answerText] = true

		results = append(results, Document{
			Text:     doc.Text,
			Metadata: copyMap(doc.Metadata),
			Score:    float64(distances[i]),
		})
		if len(results) >= topK {
			break
		}
	}

	return results, nil
}

// BatchSearch searches for multiple query embeddings in a single call.
func (r *Retriever) BatchSearch(queryEmbeddings [][]float32, topK int) ([][]Document, error) {
	if len(queryEmbeddings) == 0 {
		return nil, nil
	}
	for _, qe := range queryEmbeddings {
		if len(qe) != r.dimension {
			return nil, fmt.Errorf("query dim=%d != index dim=%d", len(qe), r.dimension)
		}
	}

	flat := make([]float32, 0, len(queryEmbeddings)*r.dimension)
	for _, qe := range queryEmbeddings {
		flat = append(flat, qe...)
	}

	searchK := topK * 5
	if searchK > len(r.documents) {
		searchK = len(r.documents)
	}

	distances, labels, err := r.index.Search(flat, int64(searchK))
	if err != nil {
		return nil, fmt.Errorf("faiss batch search: %w", err)
	}

	nq := len(queryEmbeddings)
	allResults := make([][]Document, nq)
	for q := 0; q < nq; q++ {
		var results []Document
		seen := make(map[string]bool)
		for k := 0; k < searchK; k++ {
			i := q*searchK + k
			idx := labels[i]
			if idx < 0 || int(idx) >= len(r.documents) {
				continue
			}
			doc := r.documents[idx]

			answerText := doc.Text
			if ans, ok := doc.Metadata["answer"]; ok {
				if s, ok := ans.(string); ok {
					answerText = s
				}
			}
			if seen[answerText] {
				continue
			}
			seen[answerText] = true

			results = append(results, Document{
				Text:     doc.Text,
				Metadata: copyMap(doc.Metadata),
				Score:    float64(distances[i]),
			})
			if len(results) >= topK {
				break
			}
		}
		allResults[q] = results
	}

	return allResults, nil
}

func (r *Retriever) NumDocuments() int {
	return len(r.documents)
}

func (r *Retriever) Dimension() int {
	return r.dimension
}

func (r *Retriever) Close() {
	if r.index != nil {
		r.index.Delete()
		r.index = nil
	}
}

func copyMap(src map[string]any) map[string]any {
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
