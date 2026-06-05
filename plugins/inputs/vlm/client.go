// Package vlm provides vision-language-model input sensors. They capture video
// frames (from a local camera or an RTSP stream via internal/vlm), send each
// frame to an OpenAI-compatible vision chat-completions endpoint, and surface
// the model's text descriptions as inputs.Sensor readings.
//
// It is the Go port of the Python `VLMOpenAIProvider` + `VLMOpenAI` input.
package vlm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/metrics"
)

type visionClient struct {
	name      string
	apiKey    string
	baseURL   string
	model     string
	prompt    string
	maxTokens int
	log       *zap.Logger
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (c *visionClient) describe(ctx context.Context, jpegBase64 string) (string, error) {
	requestBody := map[string]any{
		"model":      c.model,
		"max_tokens": c.maxTokens,
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": c.prompt},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url":    "data:image/jpeg;base64," + jpegBase64,
							"detail": "low",
						},
					},
				},
			},
		},
	}

	requestBytes, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/chat/completions", bytes.NewReader(requestBytes))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	start := time.Now()
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return "", fmt.Errorf("http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	metrics.RecordResponseLatency(metrics.VLMLatency, metrics.VLMLatencyLast,
		c.name, c.model, c.baseURL, req, resp, start)

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("api %d: %s", resp.StatusCode, body)
	}

	var parsed chatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	if len(parsed.Choices) == 0 {
		return "", nil
	}

	content := parsed.Choices[0].Message.Content
	c.log.Debug("Vision client response", zap.String("content", content))

	return content, nil
}
