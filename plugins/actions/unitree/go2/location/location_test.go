package location

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/providers/tts"
)

func testTTS() *tts.ElevenLabsProvider {
	return tts.ElevenLabs(tts.ElevenLabsConfig{
		OutputFormat: tts.DefaultOutputFormat,
		Rate:         tts.DefaultRate,
	}, zap.NewNop())
}

func newTestConnector(baseURL string) *Connector {
	return &Connector{
		log:     zap.NewNop(),
		tts:     testTTS(),
		baseURL: baseURL,
		apiKey:  "test-key",
		timeout: 5 * time.Second,
		mapName: "map",
	}
}

func TestParseConfigDefaults(t *testing.T) {
	cfg := parseConfig(nil)

	assert.Equal(t, localBaseURL, cfg.BaseURL, "no use_sim → local base URL")
	assert.Equal(t, defaultMapName, cfg.MapName)
	assert.Equal(t, tts.DefaultVoiceID, cfg.VoiceID)
	assert.Equal(t, tts.DefaultModelID, cfg.ModelID)
	assert.Equal(t, tts.DefaultOutputFormat, cfg.OutputFormat)
	assert.Equal(t, tts.DefaultRate, cfg.Rate)
}

func TestParseConfigUseSimBaseURL(t *testing.T) {
	cfg := parseConfig(map[string]any{"use_sim": true})
	assert.Equal(t, simBaseURL, cfg.BaseURL, "use_sim with no base_url → simulation base URL")
}

func TestParseConfigExplicitValuesPreserved(t *testing.T) {
	cfg := parseConfig(map[string]any{
		"base_url": "http://example.com/add",
		"use_sim":  true,
		"map_name": "warehouse",
		"voice_id": "v-custom",
		"rate":     8000,
	})

	assert.Equal(t, "http://example.com/add", cfg.BaseURL, "explicit base_url overrides use_sim default")
	assert.Equal(t, "warehouse", cfg.MapName)
	assert.Equal(t, "v-custom", cfg.VoiceID)
	assert.Equal(t, 8000, cfg.Rate)
}

func TestConnectPostsLocation(t *testing.T) {
	var (
		gotMethod  string
		gotAPIKey  string
		gotPayload map[string]any
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotAPIKey = r.Header.Get("x-api-key")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotPayload)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL)

	out, err := c.Connect(context.Background(), map[string]any{
		"action":      "kitchen",
		"description": "near the fridge",
	})

	require.NoError(t, err)
	assert.Nil(t, out)
	assert.Equal(t, http.MethodPost, gotMethod)
	assert.Equal(t, "test-key", gotAPIKey)
	assert.Equal(t, "map", gotPayload["map_name"])
	assert.Equal(t, "kitchen", gotPayload["label"])
	assert.Equal(t, "near the fridge", gotPayload["description"])
}

func TestConnectNon2xxIsLoggedNotReturned(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`boom`))
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL)

	out, err := c.Connect(context.Background(), map[string]any{"action": "office"})

	require.NoError(t, err, "an API error is logged, not returned")
	assert.Nil(t, out)
}

func TestConnectEmptyActionNoRequest(t *testing.T) {
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hit = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL)

	out, err := c.Connect(context.Background(), map[string]any{"action": ""})

	require.NoError(t, err)
	assert.Nil(t, out)
	assert.False(t, hit, "empty action must not trigger an HTTP request")
}

func TestConnectWrongInputType(t *testing.T) {
	c := newTestConnector("http://unused")

	out, err := c.Connect(context.Background(), "not-a-map")

	require.Error(t, err)
	assert.Nil(t, out)
}

func TestConnectRequestErrorIsSwallowed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	c := newTestConnector(url)

	out, err := c.Connect(context.Background(), map[string]any{"action": "garage"})

	require.NoError(t, err, "transport errors are logged, not returned")
	assert.Nil(t, out)
}

func TestConnectorImplementsInterface(t *testing.T) {
	var _ actions.Connector = (*Connector)(nil)
}
