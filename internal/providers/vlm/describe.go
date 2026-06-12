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
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
)

type Describer struct {
	Name      string
	APIKey    string
	BaseURL   string
	Model     string
	Prompt    string
	MaxTokens int
	Detail string
	Log    *zap.Logger
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// NewDescriber constructs a Describer, defaulting the logger when none is given.
func NewDescriber(d Describer) *Describer {
	if d.Log == nil {
		d.Log = logger.Get()
	}
	return &d
}

// Describe sends the prompt with an optional single frame; empty jpegBase64 is text-only.
func (d *Describer) Describe(ctx context.Context, jpegBase64 string) (string, error) {
	var imgs []string
	if jpegBase64 != "" {
		imgs = []string{jpegBase64}
	}
	return d.DescribeImages(ctx, imgs)
}

// DescribeImages attaches multiple frames to one request; empty entries are skipped.
func (d *Describer) DescribeImages(ctx context.Context, jpegBase64s []string) (string, error) {
	detail := d.Detail
	if detail == "" {
		detail = "low"
	}

	content := []any{
		map[string]any{"type": "text", "text": d.Prompt},
	}
	for _, jpegBase64 := range jpegBase64s {
		if jpegBase64 == "" {
			continue
		}
		content = append(content, map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url":    "data:image/jpeg;base64," + jpegBase64,
				"detail": detail,
			},
		})
	}

	requestBody := map[string]any{
		"model":      d.Model,
		"max_tokens": d.MaxTokens,
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": content,
			},
		},
	}

	requestBytes, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		d.BaseURL+"/chat/completions", bytes.NewReader(requestBytes))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+d.APIKey)

	start := time.Now()
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return "", fmt.Errorf("http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	metrics.RecordResponseLatency(metrics.VLMLatency, metrics.VLMLatencyLast,
		d.Name, d.Model, d.BaseURL, req, resp, start)

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

	result := parsed.Choices[0].Message.Content
	d.Log.Debug("Vision client response", zap.String("content", result))

	return result, nil
}
