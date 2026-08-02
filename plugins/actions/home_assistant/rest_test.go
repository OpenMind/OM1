package home_assistant

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

// capturedRequest records the last request received by the mock HA server.
type capturedRequest struct {
	path string
	body map[string]any
}

func newMockHAServer(t *testing.T, statusCode int, capture *capturedRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if capture != nil {
			capture.path = r.URL.Path
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			capture.body = body
		}
		w.WriteHeader(statusCode)
		_, _ = w.Write([]byte(`[]`))
	}))
}

func buildRESTConnector(t *testing.T, baseURL string) *RESTConnector {
	t.Helper()
	conn, err := NewRESTConnector(map[string]any{
		"base_url": baseURL,
		"token":    "test-token",
	})
	require.NoError(t, err)
	c, ok := conn.(*RESTConnector)
	require.True(t, ok, "expected *RESTConnector")
	return c
}

func TestRESTConnectorInit(t *testing.T) {
	t.Run("warns but does not fail when base_url missing", func(t *testing.T) {
		conn, err := NewRESTConnector(map[string]any{"token": "test-token"})
		require.NoError(t, err)
		require.NotNil(t, conn)
	})

	t.Run("warns but does not fail when token missing", func(t *testing.T) {
		conn, err := NewRESTConnector(map[string]any{"base_url": "http://example.com"})
		require.NoError(t, err)
		require.NotNil(t, conn)
	})

	t.Run("no-op call when base_url and token both missing", func(t *testing.T) {
		conn, err := NewRESTConnector(map[string]any{})
		require.NoError(t, err)
		c := conn.(*RESTConnector)
		_, err = c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_on",
		})
		require.NoError(t, err)
	})
}

func TestRESTConnectLight(t *testing.T) {
	t.Run("turn_on", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/light/turn_on", captured.path)
		require.Equal(t, "light.bed_light", captured.body["entity_id"])
	})

	t.Run("turn_off", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_off",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/light/turn_off", captured.path)
	})

	t.Run("set_brightness uses direct 0-255 value", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_brightness",
			"brightness":  "128",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/light/turn_on", captured.path)
		require.InDelta(t, 128, captured.body["brightness"], 0.01)
	})

	t.Run("set_brightness clamps out-of-range value", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_brightness",
			"brightness":  "999",
		})
		require.NoError(t, err)
		require.InDelta(t, 255, captured.body["brightness"], 0.01)
	})

	t.Run("set_brightness empty string falls back to default 255", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_brightness",
			"brightness":  "",
		})
		require.NoError(t, err)
		require.InDelta(t, defaultRESTBrightness, captured.body["brightness"], 0.01)
	})

	t.Run("set_color known color uses hs_color", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_color",
			"color":       "red",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/light/turn_on", captured.path)
		hs, ok := captured.body["hs_color"].([]any)
		require.True(t, ok)
		require.EqualValues(t, 0, hs[0])
		require.EqualValues(t, 100, hs[1])
	})

	t.Run("set_color multi-word name", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_color",
			"color":       "warm white",
		})
		require.NoError(t, err)
		hs, ok := captured.body["hs_color"].([]any)
		require.True(t, ok)
		require.EqualValues(t, 30, hs[0])
		require.EqualValues(t, 20, hs[1])
	})

	t.Run("set_color unknown defaults to white", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_color",
			"color":       "nonexistent",
		})
		require.NoError(t, err)
		hs, ok := captured.body["hs_color"].([]any)
		require.True(t, ok)
		require.EqualValues(t, 0, hs[0])
		require.EqualValues(t, 0, hs[1])
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "dance",
		})
		require.NoError(t, err)
		require.Empty(t, captured.path)
	})
}

func TestRESTConnectSwitch(t *testing.T) {
	t.Run("turn_on", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "switch.living_room",
			"device_type": "switch",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/switch/turn_on", captured.path)
	})

	t.Run("turn_off", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "switch.living_room",
			"device_type": "switch",
			"action":      "turn_off",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/switch/turn_off", captured.path)
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "switch.living_room",
			"device_type": "switch",
			"action":      "set_brightness",
		})
		require.NoError(t, err)
		require.Empty(t, captured.path)
	})
}

func TestRESTConnectClimate(t *testing.T) {
	t.Run("set_temperature", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.living_room",
			"device_type": "climate",
			"action":      "set_temperature",
			"temperature": "23.5",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/climate/set_temperature", captured.path)
		require.InDelta(t, 23.5, captured.body["temperature"], 0.01)
	})

	t.Run("set_temperature empty string falls back to REST default 22.0", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.living_room",
			"device_type": "climate",
			"action":      "set_temperature",
			"temperature": "",
		})
		require.NoError(t, err)
		require.InDelta(t, defaultRESTTemperature, captured.body["temperature"], 0.01)
	})

	t.Run("turn_on", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.living_room",
			"device_type": "climate",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/climate/turn_on", captured.path)
	})

	t.Run("turn_off", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.living_room",
			"device_type": "climate",
			"action":      "turn_off",
		})
		require.NoError(t, err)
		require.Equal(t, "/api/services/climate/turn_off", captured.path)
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		var captured capturedRequest
		server := newMockHAServer(t, http.StatusOK, &captured)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.living_room",
			"device_type": "climate",
			"action":      "cool_mode",
		})
		require.NoError(t, err)
		require.Empty(t, captured.path)
	})
}

func TestRESTConnectUnsupportedDeviceType(t *testing.T) {
	var captured capturedRequest
	server := newMockHAServer(t, http.StatusOK, &captured)
	defer server.Close()

	c := buildRESTConnector(t, server.URL)
	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "vacuum.robot",
		"device_type": "vacuum",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Empty(t, captured.path)
}

func TestRESTConnectMissingEntityID(t *testing.T) {
	var captured capturedRequest
	server := newMockHAServer(t, http.StatusOK, &captured)
	defer server.Close()

	c := buildRESTConnector(t, server.URL)
	_, err := c.Connect(context.Background(), map[string]any{
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Empty(t, captured.path)
}

func TestRESTNetworkErrors(t *testing.T) {
	t.Run("handles non-2xx status", func(t *testing.T) {
		server := newMockHAServer(t, http.StatusInternalServerError, nil)
		defer server.Close()

		c := buildRESTConnector(t, server.URL)
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_on",
		})
		require.Error(t, err)
	})

	t.Run("handles unreachable server", func(t *testing.T) {
		c := buildRESTConnector(t, "http://127.0.0.1:1")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_on",
		})
		require.Error(t, err)
	})
}

func TestRESTConnectorInvalidInputType(t *testing.T) {
	c := buildRESTConnector(t, "http://example.com")
	_, err := c.Connect(context.Background(), "not-a-map")
	require.Error(t, err)
}

func TestRESTConnectorTickAndStop(t *testing.T) {
	c := buildRESTConnector(t, "http://example.com")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.Tick(ctx) // should return immediately since ctx is already cancelled

	c.Stop() // no-op, should not panic
}
