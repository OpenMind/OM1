package vlm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	video "github.com/openmind/om1/internal/providers/vlm"
)

type fakeSource struct {
	frames  []video.Frame
	started chan struct{}
	stopped chan struct{}
}

func newFakeSource(frames ...video.Frame) *fakeSource {
	return &fakeSource{
		frames:  frames,
		started: make(chan struct{}, 1),
		stopped: make(chan struct{}, 1),
	}
}

func (f *fakeSource) Start(ctx context.Context) <-chan video.Frame {
	out := make(chan video.Frame)
	select {
	case f.started <- struct{}{}:
	default:
	}
	go func() {
		defer close(out)
		for _, fr := range f.frames {
			select {
			case out <- fr:
			case <-ctx.Done():
				return
			}
		}
		<-ctx.Done()
	}()
	return out
}

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

func TestParseConfigGeminiDefaults(t *testing.T) {
	cfg, err := parseConfig(map[string]any{"api_key": "k"}, geminiDefaults)
	require.NoError(t, err)
	assert.Equal(t, geminiDefaults.baseURL, cfg.BaseURL)
	assert.Equal(t, "gemini-2.5-flash", cfg.Model)
	assert.Equal(t, geminiDefaults.maxTokens, cfg.MaxTokens)
}

func TestParseConfigOverrides(t *testing.T) {
	cfg, err := parseConfig(map[string]any{
		"api_key": "k",
		"model":   "custom-model",
		"prompt":  "custom prompt",
	}, geminiDefaults)
	require.NoError(t, err)
	assert.Equal(t, "custom-model", cfg.Model)
	assert.Equal(t, "custom prompt", cfg.Prompt)
}

func TestParseConfigRequiresAPIKey(t *testing.T) {
	_, err := parseConfig(map[string]any{}, openAIDefaults)
	require.Error(t, err)
}

func TestGeminiSensorsRegistered(t *testing.T) {
	for _, name := range []string{"VLMOpenAI", "VLMOpenAIRTSP", "VLMGemini", "VLMGeminiRTSP"} {
		s, err := inputs.Load(name, map[string]any{"api_key": "k", "rtsp_url": "rtsp://x"})
		require.NoError(t, err, name)
		require.NotNil(t, s, name)
		s.Stop()
	}
}

func TestSensorListenDescribesFramesAndBuffers(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content []struct {
					Type     string `json:"type"`
					ImageURL struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		require.NotEmpty(t, body.Messages)
		assert.Contains(t, body.Messages[0].Content[1].ImageURL.URL, "data:image/jpeg;base64,")

		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"a robot"}}]}`))
	}))
	defer srv.Close()

	source := newFakeSource(video.Frame{Timestamp: time.Unix(1, 0), JPEG: []byte{0x01, 0x02}})
	s := NewSensor("VLMOpenAI", VLMConfig{
		APIKey:  "k",
		BaseURL: srv.URL,
		Model:   "test",
		Prompt:  "describe",
	}, source)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	out, err := s.Listen(ctx)
	require.NoError(t, err)

	select {
	case raw := <-out:
		assert.Equal(t, "a robot", raw)
		_, err := s.RawToText(ctx, raw)
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for description")
	}

	formatted := s.FormattedLatestBuffer()
	assert.Contains(t, formatted, "Vision")
	assert.Contains(t, formatted, "a robot")

	assert.Equal(t, "", s.FormattedLatestBuffer())

	s.Stop()
	select {
	case <-source.stopped:
	case <-time.After(time.Second):
		t.Fatal("source was not stopped")
	}
}

func TestRawToTextIgnoresNonStrings(t *testing.T) {
	s := NewSensor("VLMOpenAI", VLMConfig{APIKey: "k", BaseURL: "http://x"}, newFakeSource())
	msg, err := s.RawToText(context.Background(), 123)
	require.NoError(t, err)
	assert.Nil(t, msg)
	assert.Equal(t, "", s.FormattedLatestBuffer())
}

func TestRawToTextBoundsHistory(t *testing.T) {
	s := &vlmSensor{name: "VLMOpenAI", log: zap.NewNop()}
	for i := 0; i < vlmMaxMessages+5; i++ {
		_, _ = s.RawToText(context.Background(), "msg")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	assert.Len(t, s.messages, vlmMaxMessages)
}
