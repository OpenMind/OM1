package qualityscore

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/tracer/tracetype"
)

func TestExtractPrompt(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"double quotes", `System prompt...\nVoice: "hello there"\nmore context`, "hello there"},
		{"single quotes", `Voice: 'hi robot'`, "hi robot"},
		{"spans newlines", "Voice: \"line one\nline two\"", "line one\nline two"},
		{"no marker", "just some text with no marker", ""},
		{"trims whitespace", `Voice:   "  padded  "  `, "padded"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractPrompt(tc.input); got != tc.want {
				t.Errorf("extractPrompt(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestExtractResponse(t *testing.T) {
	cases := []struct {
		name         string
		output       []map[string]any
		wantResponse string
		wantType     string
	}{
		{
			name: "greeting_conversation",
			output: []map[string]any{
				{"type": "greeting_conversation", "value": map[string]any{"response": " hi there "}},
			},
			wantResponse: "hi there",
			wantType:     "greeting_conversation",
		},
		{
			name: "speak",
			output: []map[string]any{
				{"type": "emotion", "value": map[string]any{"emotion": "happy"}},
				{"type": "speak", "value": map[string]any{"action": "hello!"}},
			},
			wantResponse: "hello!",
			wantType:     "speak",
		},
		{
			name:         "no spoken action",
			output:       []map[string]any{{"type": "robot_action", "value": map[string]any{"action": "wave"}}},
			wantResponse: "",
			wantType:     "",
		},
		{
			name:         "empty",
			output:       nil,
			wantResponse: "",
			wantType:     "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, typ := extractResponse(tc.output)
			if resp != tc.wantResponse || typ != tc.wantType {
				t.Errorf("extractResponse() = (%q, %q), want (%q, %q)", resp, typ, tc.wantResponse, tc.wantType)
			}
		})
	}
}

func TestDetectLang(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		{"english stopword", "hello can you hear me", "en"},
		{"korean hangul", "안녕하세요 로봇", "ko"},
		{"japanese kana", "こんにちは", "ja"},
		{"chinese han", "你好机器人今天天气怎么样", "zh-cn"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := detectLang(tc.text); got != tc.want {
				t.Errorf("detectLang(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

func TestLangName(t *testing.T) {
	if got := langName("en"); got != "English" {
		t.Errorf("langName(en) = %q, want English", got)
	}
	if got := langName("xx-unknown"); got != "xx-unknown" {
		t.Errorf("langName(unknown) should pass through unchanged, got %q", got)
	}
}

func TestInitLanguageLabels(t *testing.T) {
	initLanguageLabels()

	require.Equal(t, 0.0, testutil.ToFloat64(metrics.QualityLiveLanguageCount.WithLabelValues("Spanish")),
		"pre-registered at zero -- gives increase() a real prior sample before the language's first real occurrence")

	metrics.QualityLiveLanguageCount.WithLabelValues("Spanish").Inc()
	initLanguageLabels()
	require.Equal(t, 1.0, testutil.ToFloat64(metrics.QualityLiveLanguageCount.WithLabelValues("Spanish")),
		"calling initLanguageLabels again must not reset an already-incremented language")
}

func TestEndToEnd(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req chatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		label := "positive"
		if strings.Contains(req.Messages[0].Content, "coherent") {
			label = "coherent"
		}
		resp := chatCompletionResponse{}
		content, _ := json.Marshal(labelResult{Label: label})
		resp.Choices = []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}{{Message: struct {
			Content string `json:"content"`
		}{Content: string(content)}}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	oldWd, _ := os.Getwd()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("chdir: %v", err)
	}
	defer func() { _ = os.Chdir(oldWd) }()

	records := make(chan tracetype.TraceRecord, 1)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	Start(ctx, config.QualityScorerConfig{
		Enabled: true,
		BaseURL: server.URL,
		APIKey:  "test-key",
	}, records, zap.NewNop())

	records <- tracetype.TraceRecord{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		LLMInput:  `context...\nVoice: "hello there robot"\nmore`,
		LLMOutput: []map[string]any{
			{"type": "speak", "value": map[string]any{"action": "hi! nice to meet you"}},
		},
	}

	deadline := time.After(2 * time.Second)
	for {
		families, err := prometheus.DefaultGatherer.Gather()
		if err != nil {
			t.Fatalf("gather: %v", err)
		}
		found := false
		for _, f := range families {
			if f.GetName() == "om1_quality_live_turns_scored" && len(f.GetMetric()) > 0 && f.GetMetric()[0].GetCounter().GetValue() > 0 {
				found = true
			}
		}
		if found {
			break
		}
		select {
		case <-deadline:
			t.Fatal("timed out waiting for the scored turn to appear in metrics")
		case <-time.After(10 * time.Millisecond):
		}
	}

	deadline = time.After(2 * time.Second)
	var (
		logBytes []byte
		err      error
	)
	for {
		logBytes, err = os.ReadFile(filepath.Join(tmpDir, logPathForNow()))
		if err == nil && strings.Contains(string(logBytes), "hello there robot") {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for classification log write, last read: %v, %s", err, logBytes)
		case <-time.After(10 * time.Millisecond):
		}
	}

	cancel()

	if callCount != 2 {
		t.Errorf("expected 2 classification calls (input + coherence), got %d", callCount)
	}
}
