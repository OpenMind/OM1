package knowledgebase

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

type HTTPEmbedder struct {
	baseURL string
	client  *http.Client
}

func NewHTTPEmbedder(baseURL string) *HTTPEmbedder {
	return &HTTPEmbedder{
		baseURL: baseURL,
		client:  httpclient.Default(),
	}
}

type embedRequest struct {
	Query string `json:"query"`
}

type embedResponse struct {
	EmbeddingB64 string  `json:"embedding_b64"`
	LatencyMs    float64 `json:"latency_ms"`
}

func (e *HTTPEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(embedRequest{Query: text})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	ctx2, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req = req.WithContext(ctx2)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed server returned %d", resp.StatusCode)
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	raw, err := base64.StdEncoding.DecodeString(result.EmbeddingB64)
	if err != nil {
		return nil, fmt.Errorf("decode base64 embedding: %w", err)
	}

	return bytesToFloat32(raw)
}

type embedBatchRequest struct {
	Queries []string `json:"queries"`
}

type embedBatchResponse struct {
	EmbeddingsB64 []string `json:"embeddings_b64"`
	LatencyMs     float64  `json:"latency_ms"`
}

func (e *HTTPEmbedder) EmbedBatch(ctx context.Context, queries []string) ([][]float32, error) {
	body, err := json.Marshal(embedBatchRequest{Queries: queries})
	if err != nil {
		return nil, fmt.Errorf("marshal embed_batch request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embed_batch", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed_batch request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	ctx2, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req = req.WithContext(ctx2)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed_batch request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed server returned %d", resp.StatusCode)
	}

	var result embedBatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed_batch response: %w", err)
	}

	embeddings := make([][]float32, len(result.EmbeddingsB64))
	for i, embB64 := range result.EmbeddingsB64 {
		raw, err := base64.StdEncoding.DecodeString(embB64)
		if err != nil {
			return nil, fmt.Errorf("decode base64 embedding[%d]: %w", i, err)
		}
		vec, err := bytesToFloat32(raw)
		if err != nil {
			return nil, fmt.Errorf("parse embedding[%d]: %w", i, err)
		}
		embeddings[i] = vec
	}

	return embeddings, nil
}

func bytesToFloat32(raw []byte) ([]float32, error) {
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("byte length %d not divisible by 4", len(raw))
	}
	n := len(raw) / 4
	out := make([]float32, n)
	reader := bytes.NewReader(raw)
	if err := binary.Read(reader, binary.LittleEndian, &out); err != nil {
		return nil, fmt.Errorf("binary read float32: %w", err)
	}
	return out, nil
}
