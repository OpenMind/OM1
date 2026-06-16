package vlm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/providers"
	video "github.com/openmind/om1/internal/providers/vlm"
)

type fakeSource struct {
	ch      chan video.Frame
	stopped chan struct{}
}

func newFakeSource() *fakeSource {
	return &fakeSource{ch: make(chan video.Frame, 1), stopped: make(chan struct{}, 1)}
}

func (f *fakeSource) Start(context.Context) <-chan video.Frame { return f.ch }

func (f *fakeSource) Stop() {
	select {
	case f.stopped <- struct{}{}:
	default:
	}
}

func TestParseConfigDefaults(t *testing.T) {
	cfg, err := parseConfig(map[string]any{"api_key": "k"}, openAIDefaults)
	require.NoError(t, err)
	assert.Equal(t, openAIDefaults.baseURL, cfg.BaseURL)
	assert.Equal(t, openAIDefaults.model, cfg.Model)
	assert.Equal(t, openAIDefaults.prompt, cfg.Prompt)
	assert.Equal(t, defaultFPS, cfg.FPS)
	assert.Equal(t, openAIDefaults.maxTokens, cfg.MaxTokens)
}

func TestParseConfigRequiresAPIKey(t *testing.T) {
	_, err := parseConfig(map[string]any{}, geminiDefaults)
	require.Error(t, err)
}

func TestParseConfigOverrides(t *testing.T) {
	cfg, err := parseConfig(map[string]any{
		"api_key": "k",
		"model":   "custom-model",
		"prompt":  "custom prompt",
		"fps":     4,
	}, geminiDefaults)
	require.NoError(t, err)
	assert.Equal(t, "custom-model", cfg.Model)
	assert.Equal(t, "custom prompt", cfg.Prompt)
	assert.Equal(t, 4, cfg.FPS)
}

func TestBackgroundsRegistered(t *testing.T) {
	for _, name := range []string{"VLMOpenAI", "VLMOpenAIRTSP", "VLMGemini", "VLMGeminiRTSP"} {
		b, err := bg.Load(name, map[string]any{"api_key": "k", "rtsp_url": "rtsp://x"})
		require.NoError(t, err, name)
		require.NotNil(t, b, name)
		b.Stop()
	}
}

func TestRunPublishesFrameAndDescription(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"a cat on a desk"}}]}`))
	}))
	defer srv.Close()

	source := newFakeSource()
	b := NewBackground("test", VLMConfig{
		APIKey:    "k",
		BaseURL:   srv.URL,
		Model:     "test",
		Prompt:    "describe",
		MaxTokens: 16,
	}, source)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go b.Run(ctx)

	ts := time.Unix(1234, 0)
	source.ch <- video.Frame{Timestamp: ts, JPEG: []byte{0x01, 0x02}}

	require.Eventually(t, func() bool {
		text, _, ok := video.LatestDescription().Get()
		return ok && text == "a cat on a desk"
	}, 2*time.Second, 10*time.Millisecond, "description should be published")

	jpeg, _, ok := providers.LatestFrame().Get()
	require.True(t, ok, "frame should be published")
	assert.Equal(t, []byte{0x01, 0x02}, jpeg)

	cancel()
	b.Stop()
	select {
	case <-source.stopped:
	case <-time.After(time.Second):
		t.Fatal("source was not stopped")
	}
}
