package knowledgebase

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

// DefaultBaseURL is the embedding service endpoint used when none is configured.
const DefaultBaseURL = "http://10.1.10.221:8100"

// embedTimeout bounds a single embedding request, matching the Python client.
const embedTimeout = 30 * time.Second

// Embedder converts query text into an embedding vector.
type Embedder interface {
	Embed(ctx context.Context, query string) ([]float32, error)
}

// HTTPEmbedder is an Embedder that sends queries to an external embedding service.
type HTTPEmbedder struct {
	baseURL string
	client  *http.Client
}

// NewHTTPEmbedder creates an embedder pointing at baseURL.
func NewHTTPEmbedder(baseURL string) *HTTPEmbedder {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &HTTPEmbedder{
		baseURL: baseURL,
		client:  httpclient.Default(),
	}
}

// embedResponse is the JSON shape returned by the /embed endpoint.
type embedResponse struct {
	EmbeddingB64 string `json:"embedding_b64"`
}

// Embed returns the embedding vector for a single query string.
func (e *HTTPEmbedder) Embed(ctx context.Context, query string) ([]float32, error) {
	body, err := json.Marshal(map[string]string{"query": query})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, embedTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("embed service returned %s: %s", resp.Status, snippet)
	}

	var decoded embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	raw, err := base64.StdEncoding.DecodeString(decoded.EmbeddingB64)
	if err != nil {
		return nil, fmt.Errorf("base64 decode embedding: %w", err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("embedding byte length %d is not a multiple of 4", len(raw))
	}

	vec := make([]float32, len(raw)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
		vec[i] = math.Float32frombits(bits)
	}
	return vec, nil
}
