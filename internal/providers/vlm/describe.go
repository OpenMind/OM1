package vlm

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
	Log       *zap.Logger
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

const (
	historyHeader = "--- Memory: earlier camera frames and what you reported for each, " +
		"oldest first. They are context for how the scene got here; do not " +
		"describe them as if they were happening now. ---"
	textCacheHeader = "Other recent reports (frames omitted):"
	currentHeader   = "--- Current camera frame (now) ---"
	historyFooter   = "Answer for the current frame, using the memory above as context."
)

// Describe sends the prompt to the vision endpoint with a single frame and no
// memory. See DescribeWithHistory for the general form.
func (d *Describer) Describe(ctx context.Context, jpegBase64 string) (string, error) {
	return d.DescribeWithHistory(ctx, jpegBase64, History{})
}

// DescribeWithHistory sends the prompt, any recalled history, and the current
// frame to the vision endpoint and returns the generated text. When jpegBase64
// is non-empty the frame is attached as an image; when it is empty the request
// carries no current frame, so callers can still get a response if frame
// capture failed. An empty result is returned (without error) when the model
// produces no choices.
func (d *Describer) DescribeWithHistory(ctx context.Context, jpegBase64 string, hist History) (string, error) {
	content := buildContent(d.Prompt, jpegBase64, hist)

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

// buildContent lays out the multimodal user message: the prompt, then the
// recalled history interleaved as description/frame pairs, then the current
// frame last so it is the most recent thing the model sees.
func buildContent(prompt, jpegBase64 string, hist History) []map[string]any {
	content := []map[string]any{textPart(prompt)}

	if !hist.Empty() {
		content = append(content, textPart(historyHeader))
		for _, step := range hist.Frames {
			content = append(content,
				textPart(fmt.Sprintf("[%s ago] you reported: %q", formatAge(step.Age), step.Description)),
				imagePart(step.JPEGBase64),
			)
		}
		if len(hist.Texts) > 0 {
			var b strings.Builder
			b.WriteString(textCacheHeader)
			for _, step := range hist.Texts {
				fmt.Fprintf(&b, "\n- [%s ago] %q", formatAge(step.Age), step.Description)
			}
			content = append(content, textPart(b.String()))
		}
		content = append(content, textPart(currentHeader))
	}

	if jpegBase64 != "" {
		content = append(content, imagePart(jpegBase64))
	}

	if !hist.Empty() {
		content = append(content, textPart(historyFooter))
	}

	return content
}

func textPart(text string) map[string]any {
	return map[string]any{"type": "text", "text": text}
}

func imagePart(jpegBase64 string) map[string]any {
	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url":    "data:image/jpeg;base64," + jpegBase64,
			"detail": "low",
		},
	}
}

// formatAge renders a step age compactly for the prompt.
func formatAge(age time.Duration) string {
	if age < time.Minute {
		return fmt.Sprintf("%.1fs", age.Seconds())
	}
	return age.Round(time.Second).String()
}
