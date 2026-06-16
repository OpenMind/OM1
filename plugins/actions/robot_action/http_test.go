package robot_action

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
)

func newTestConnector(baseURL string, override map[string]ActionMapping) *Connector {
	mappings := make(map[string]ActionMapping, len(defaultActions)+len(override))
	for name, mapping := range defaultActions {
		mappings[name] = mapping
	}
	for name, mapping := range override {
		mappings[name] = mapping
	}
	return &Connector{
		log:     zap.NewNop(),
		baseURL: strings.TrimRight(baseURL, "/"),
		timeout: 5 * time.Second,
		actions: mappings,
	}
}

func TestRobotActionEnumIncludesSupportedActions(t *testing.T) {
	values := RobotAction("").EnumValues()
	assert.Contains(t, values, "wave")
	assert.Contains(t, values, "shakehand")
}

func TestDefaultActionsHandshakeMappings(t *testing.T) {
	mapping, ok := defaultActions["wave"]
	require.True(t, ok, "wave should have a built-in mapping")
	assert.Equal(t, "/motion", mapping.Path)
	assert.Equal(t, "wave", mapping.Body["motion"])

	mapping, ok = defaultActions["shakehand"]
	require.True(t, ok, "shakehand should have a built-in mapping")
	assert.Equal(t, "/motion", mapping.Path)
	assert.Equal(t, "handshake", mapping.Body["motion"])
}

func TestNewHTTPConnectorAppliesDefaults(t *testing.T) {
	c, err := NewHTTPConnector(nil)
	require.NoError(t, err)

	conn, ok := c.(*Connector)
	require.True(t, ok)
	assert.Equal(t, defaultBaseURL, conn.baseURL, "missing base_url falls back to the configured default")
	assert.Equal(t, defaultTimeout, conn.timeout)
	assert.Equal(t, "/motion", conn.actions["wave"].Path, "built-in wave mapping is loaded by default")
	assert.Equal(t, "/motion", conn.actions["shakehand"].Path, "built-in shakehand mapping is loaded by default")
}

func TestNewHTTPConnectorTrimsTrailingSlash(t *testing.T) {
	c, err := NewHTTPConnector(map[string]any{
		"base_url": "http://192.168.10.102:8080/",
	})
	require.NoError(t, err)
	conn := c.(*Connector)
	assert.Equal(t, "http://192.168.10.102:8080", conn.baseURL, "trailing slash is normalized away")
}

func TestNewHTTPConnectorActionOverrideMerges(t *testing.T) {
	c, err := NewHTTPConnector(map[string]any{
		"actions": map[string]any{
			"wave": map[string]any{
				"path": "/wave_override",
				"body": map[string]any{"delay": 2.0},
			},
			"sit": map[string]any{
				"path": "/sit",
			},
		},
	})
	require.NoError(t, err)

	conn := c.(*Connector)
	assert.Equal(t, "/wave_override", conn.actions["wave"].Path, "config overrides the built-in wave path")
	assert.Equal(t, 2.0, conn.actions["wave"].Body["delay"], "config overrides the built-in wave body")
	assert.Equal(t, "/sit", conn.actions["sit"].Path, "config can add new actions")
}

func TestConnectPostsMotionForWave(t *testing.T) {
	var (
		gotMethod      string
		gotPath        string
		gotContentType string
		gotPayload     map[string]any
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotContentType = r.Header.Get("Content-Type")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotPayload)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, nil)

	out, err := c.Connect(context.Background(), map[string]any{"action": "wave"})

	require.NoError(t, err)
	assert.Nil(t, out)
	assert.Equal(t, http.MethodPost, gotMethod)
	assert.Equal(t, "/motion", gotPath, "wave maps to /motion")
	assert.Equal(t, "application/json", gotContentType)
	assert.Equal(t, "wave", gotPayload["motion"])
}

func TestConnectPostsMotionForShakehand(t *testing.T) {
	var (
		gotMethod      string
		gotPath        string
		gotContentType string
		gotPayload     map[string]any
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotContentType = r.Header.Get("Content-Type")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotPayload)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, nil)

	out, err := c.Connect(context.Background(), map[string]any{"action": "shakehand"})

	require.NoError(t, err)
	assert.Nil(t, out)
	assert.Equal(t, http.MethodPost, gotMethod)
	assert.Equal(t, "/motion", gotPath, "shakehand maps to /motion")
	assert.Equal(t, "application/json", gotContentType)
	assert.Equal(t, "handshake", gotPayload["motion"])
}

func TestConnectOverrideMappingIsUsed(t *testing.T) {
	var gotPath string
	var gotPayload map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &gotPayload)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, map[string]ActionMapping{
		"wave": {Path: "/wave_v2", Body: map[string]any{"motion": "wave_v2"}},
	})

	_, err := c.Connect(context.Background(), map[string]any{"action": "wave"})
	require.NoError(t, err)
	assert.Equal(t, "/wave_v2", gotPath)
	assert.Equal(t, "wave_v2", gotPayload["motion"])
}

func TestConnectUnknownActionFallsBackToActionPath(t *testing.T) {
	var (
		gotPath   string
		gotLength int64
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotLength = r.ContentLength
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, nil)

	_, err := c.Connect(context.Background(), map[string]any{"action": "spin"})
	require.NoError(t, err)
	assert.Equal(t, "/spin", gotPath, "unmapped actions fall back to /{action}")
	assert.Equal(t, int64(0), gotLength, "fallback mapping sends no body")
}

func TestConnectEmptyActionNoRequest(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, nil)

	out, err := c.Connect(context.Background(), map[string]any{"action": "  "})
	require.NoError(t, err)
	assert.Nil(t, out)
	assert.Equal(t, int32(0), hits.Load(), "blank action must not trigger an HTTP request")
}

func TestConnectWrongInputType(t *testing.T) {
	c := newTestConnector("http://unused", nil)

	out, err := c.Connect(context.Background(), "not-a-map")

	require.Error(t, err)
	assert.Nil(t, out)
}

func TestConnectNon2xxIsLoggedNotReturned(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`boom`))
	}))
	defer srv.Close()

	c := newTestConnector(srv.URL, nil)

	out, err := c.Connect(context.Background(), map[string]any{"action": "wave"})

	require.NoError(t, err, "an API error is logged, not returned")
	assert.Nil(t, out)
}

func TestConnectRequestErrorIsSwallowed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	c := newTestConnector(url, nil)

	out, err := c.Connect(context.Background(), map[string]any{"action": "wave"})

	require.NoError(t, err, "transport errors are logged, not returned")
	assert.Nil(t, out)
}

func TestConnectorImplementsInterface(t *testing.T) {
	var _ actions.Connector = (*Connector)(nil)
}
