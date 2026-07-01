package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/knowledgebase"
)

const searchTimeout = 2 * time.Second

// Retriever searches memory via cloud API with local embedding.
type Retriever struct {
	apiURL   string
	apiKey   string
	embedder knowledgebase.Embedder
	client   *http.Client
	log      *zap.Logger
}

// NewRetriever creates a Retriever that embeds locally and searches remotely.
func NewRetriever(apiURL, apiKey, embedderURL string, log *zap.Logger) *Retriever {
	return &Retriever{
		apiURL:   apiURL,
		apiKey:   apiKey,
		embedder: knowledgebase.NewHTTPEmbedder(embedderURL),
		client:   httpclient.Default(),
		log:      log,
	}
}

type searchRequest struct {
	Embedding []float32 `json:"embedding"`
	QueryText string    `json:"query_text"`
	TopK      int       `json:"top_k"`
	MinScore  float64   `json:"min_score"`
	UserID    string    `json:"user_id,omitempty"`
}

// SearchResult holds the response.
type SearchResult struct {
	Chunks       []MemoryEntry
	Profile      *ProfileData
	FactsSummary string
}

// ProfileData mirrors the backend search profile response.
type ProfileData struct {
	Names            []string `json:"names"`
	VisitCount       int      `json:"visit_count"`
	InteractionCount int      `json:"interaction_count"`
	FirstSeen        string   `json:"first_seen"`
	LastSeen         string   `json:"last_seen"`
}

// Search embeds the query locally, sends the vector to the cloud, returns results.
func (r *Retriever) Search(ctx context.Context, query, userID string, topK int, minScore float64) (*SearchResult, error) {
	if strings.TrimSpace(query) == "" && userID == "" {
		return &SearchResult{}, nil
	}

	var embedding []float32
	if strings.TrimSpace(query) != "" {
		var err error
		embedding, err = r.embedder.Embed(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
	}

	body, _ := json.Marshal(searchRequest{
		Embedding: embedding,
		QueryText: query,
		TopK:      topK,
		MinScore:  minScore,
		UserID:    userID,
	})

	ctx, cancel := context.WithTimeout(ctx, searchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.apiURL+"/memory/search", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+r.apiKey)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("search returned %s: %s", resp.Status, snippet)
	}

	var raw struct {
		Chunks []struct {
			Text     string            `json:"text"`
			Score    float64           `json:"score"`
			Metadata map[string]string `json:"metadata"`
		} `json:"chunks"`
		Profile      *ProfileData `json:"profile"`
		FactsSummary string       `json:"facts_summary"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}

	result := &SearchResult{
		Profile:      raw.Profile,
		FactsSummary: raw.FactsSummary,
	}
	for _, c := range raw.Chunks {
		result.Chunks = append(result.Chunks, MemoryEntry{
			Text:     c.Text,
			Score:    c.Score,
			Metadata: c.Metadata,
		})
	}
	return result, nil
}
