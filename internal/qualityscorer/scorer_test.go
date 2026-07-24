package qualityscorer

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
	dto "github.com/prometheus/client_model/go"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/providers"
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

func TestLiveCollectorMetrics(t *testing.T) {
	c := newLiveCollector()
	now := time.Now()
	c.record(scoreEvent{at: now, language: "English", classification: "positive", coherence: "coherent"})
	c.record(scoreEvent{at: now, language: "English", classification: "negative", coherence: ""})
	c.record(scoreEvent{at: now.Add(-2 * time.Hour), language: "Spanish", classification: "positive", coherence: "marginal"})

	reg := prometheus.NewRegistry()
	if err := reg.Register(c); err != nil {
		t.Fatalf("register: %v", err)
	}
	families, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}

	byName := map[string]*dto.MetricFamily{}
	for _, f := range families {
		byName[f.GetName()] = f
	}

	langFamily, ok := byName["om1_quality_live_language_count"]
	if !ok {
		t.Fatal("missing om1_quality_live_language_count")
	}
	var englishAllTime float64
	for _, m := range langFamily.GetMetric() {
		for _, l := range m.GetLabel() {
			if l.GetName() == "language" && l.GetValue() == "English" {
				englishAllTime = m.GetGauge().GetValue()
			}
		}
	}
	if englishAllTime != 2 {
		t.Errorf("English all-time count = %v, want 2", englishAllTime)
	}

	// The 2-hour-old Spanish event must not appear in the trailing-1h metric.
	recentFamily, ok := byName["om1_quality_live_language_count_last_1h"]
	if !ok {
		t.Fatal("missing om1_quality_live_language_count_last_1h")
	}
	for _, m := range recentFamily.GetMetric() {
		for _, l := range m.GetLabel() {
			if l.GetName() == "language" && l.GetValue() == "Spanish" {
				t.Errorf("Spanish (2h old) should be pruned from the last-1h window, got %v", m.GetGauge().GetValue())
			}
		}
	}

	turnsFamily, ok := byName["om1_quality_live_turns_scored"]
	if !ok {
		t.Fatal("missing om1_quality_live_turns_scored")
	}
	if got := turnsFamily.GetMetric()[0].GetCounter().GetValue(); got != 2 {
		t.Errorf("turns_scored = %v, want 2 (only records with a coherence label count)", got)
	}

	activeFamily, ok := byName["om1_quality_live_active_score"]
	if !ok {
		t.Fatal("missing om1_quality_live_active_score")
	}
	if got := activeFamily.GetMetric()[0].GetGauge().GetValue(); got != 0.5 {
		t.Errorf("active_score = %v, want 0.5 (last recorded coherence label was marginal)", got)
	}
}

// TestEndToEnd wires a real Tracer through StartServer against a fake OpenAI
// endpoint, calls Gauge() the way runtime.go does on every LLM turn, and
// confirms a scored metric shows up on the registry -- exercising the exact
// seam Gauge() -> channel -> scoreOne -> classify -> Prometheus that this
// whole package exists to provide, without needing a real API key or a live
// OM1 conversation.
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

	tracer := providers.TracerProvider()
	tracer.Enable()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stop := StartServer(ctx, zap.NewNop(), tracer, config.QualityScorerConfig{
		Enabled: true,
		BaseURL: server.URL,
		APIKey:  "test-key",
	})

	tracer.Gauge(`context...\nVoice: "hello there robot"\nmore`, []map[string]any{
		{"type": "speak", "value": map[string]any{"action": "hi! nice to meet you"}},
	})

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

	cancel()
	stop()

	if callCount != 2 {
		t.Errorf("expected 2 classification calls (input + coherence), got %d", callCount)
	}

	logBytes, err := os.ReadFile(filepath.Join(tmpDir, defaultLogPath))
	if err != nil {
		t.Fatalf("read classification log: %v", err)
	}
	if !strings.Contains(string(logBytes), "hello there robot") {
		t.Errorf("classification log missing expected prompt, got: %s", logBytes)
	}
}
