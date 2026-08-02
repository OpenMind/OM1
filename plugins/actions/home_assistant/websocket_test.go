package home_assistant

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
)

// haWSServer records the received call_service command and responds with a
// scripted Home Assistant WebSocket handshake + result.
type haWSServer struct {
	server      *httptest.Server
	authOK      bool
	commandOK   bool
	lastCommand map[string]any
}

func newHAWSServer(t *testing.T, authOK, commandOK bool) *haWSServer {
	t.Helper()
	h := &haWSServer{authOK: authOK, commandOK: commandOK}

	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	h.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()

		if err := conn.WriteJSON(map[string]any{"type": "auth_required"}); err != nil {
			return
		}

		var authMsg map[string]any
		if err := conn.ReadJSON(&authMsg); err != nil {
			return
		}

		if !h.authOK {
			_ = conn.WriteJSON(map[string]any{"type": "auth_invalid"})
			return
		}
		if err := conn.WriteJSON(map[string]any{"type": "auth_ok"}); err != nil {
			return
		}

		var command map[string]any
		if err := conn.ReadJSON(&command); err != nil {
			return
		}
		h.lastCommand = command

		if h.commandOK {
			_ = conn.WriteJSON(map[string]any{"success": true})
		} else {
			_ = conn.WriteJSON(map[string]any{
				"success": false,
				"error":   map[string]any{"code": "unknown", "message": "boom"},
			})
		}
	}))
	t.Cleanup(h.server.Close)
	return h
}

func (h *haWSServer) baseURL() string {
	return "http" + strings.TrimPrefix(h.server.URL, "http")
}

func buildWSConnector(t *testing.T, baseURL string) *WSConnector {
	t.Helper()
	conn, err := NewWSConnector(map[string]any{
		"base_url": baseURL,
		"token":    "test-token",
	})
	require.NoError(t, err)
	c, ok := conn.(*WSConnector)
	require.True(t, ok, "expected *WSConnector")
	return c
}

func TestBuildWSURL(t *testing.T) {
	require.Equal(t, "ws://example.com/api/websocket", buildWSURL("http://example.com"))
	require.Equal(t, "wss://example.com/api/websocket", buildWSURL("https://example.com"))
	require.Equal(t, "ws://example.com/api/websocket", buildWSURL("http://example.com/"))
	require.Equal(t, "example.com/api/websocket", buildWSURL("example.com"))
}

func TestWSConnectorInit(t *testing.T) {
	t.Run("warns but does not fail when base_url missing", func(t *testing.T) {
		conn, err := NewWSConnector(map[string]any{"token": "test-token"})
		require.NoError(t, err)
		require.NotNil(t, conn)
	})

	t.Run("warns but does not fail when token missing", func(t *testing.T) {
		conn, err := NewWSConnector(map[string]any{"base_url": "http://example.com"})
		require.NoError(t, err)
		require.NotNil(t, conn)
	})
}

func TestWSConnectLightTurnOn(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.NotNil(t, h.lastCommand)
	require.Equal(t, "light", h.lastCommand["domain"])
	require.Equal(t, "turn_on", h.lastCommand["service"])
	target, _ := h.lastCommand["target"].(map[string]any)
	require.Equal(t, "light.bed_light", target["entity_id"])
}

func TestWSConnectLightSetBrightness(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "set_brightness",
		"brightness":  "128",
	})
	require.NoError(t, err)
	data, _ := h.lastCommand["service_data"].(map[string]any)
	require.InDelta(t, 128, data["brightness"], 0.01)
}

func TestWSConnectLightSetColor(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "set_color",
		"color":       "blue",
	})
	require.NoError(t, err)
	data, _ := h.lastCommand["service_data"].(map[string]any)
	hs, _ := data["hs_color"].([]any)
	require.EqualValues(t, 240, hs[0])
	require.EqualValues(t, 100, hs[1])
}

func TestWSConnectSwitch(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "switch.living_room",
		"device_type": "switch",
		"action":      "turn_off",
	})
	require.NoError(t, err)
	require.Equal(t, "switch", h.lastCommand["domain"])
	require.Equal(t, "turn_off", h.lastCommand["service"])
}

func TestWSConnectClimateSetTemperature(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "climate.living_room",
		"device_type": "climate",
		"action":      "set_temperature",
		"temperature": "",
	})
	require.NoError(t, err)
	data, _ := h.lastCommand["service_data"].(map[string]any)
	require.InDelta(t, defaultWSTemperature, data["temperature"], 0.01)
}

func TestWSConnectUnsupportedDeviceType(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "vacuum.robot",
		"device_type": "vacuum",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Nil(t, h.lastCommand)
}

func TestWSConnectMissingEntityID(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Nil(t, h.lastCommand)
}

func TestWSSendCommandAuthFailed(t *testing.T) {
	h := newHAWSServer(t, false, true)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err) // Connect never returns an error; failures are logged
}

func TestWSSendCommandFailedResult(t *testing.T) {
	h := newHAWSServer(t, true, false)
	c := buildWSConnector(t, h.baseURL())

	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err)
}

func TestWSSendCommandNoTokenOrURL(t *testing.T) {
	c := buildWSConnector(t, "")
	ok := c.sendCommand(context.Background(), "light", "turn_on", "light.bed_light", nil)
	require.False(t, ok)
}

func TestWSSendCommandUnreachableServer(t *testing.T) {
	c := buildWSConnector(t, "http://127.0.0.1:1")
	ok := c.sendCommand(context.Background(), "light", "turn_on", "light.bed_light", nil)
	require.False(t, ok)
}

func TestWSMsgIDIncrements(t *testing.T) {
	h := newHAWSServer(t, true, true)
	c := buildWSConnector(t, h.baseURL())

	c.sendCommand(context.Background(), "light", "turn_on", "light.bed_light", nil)
	firstID := h.lastCommand["id"]

	c.sendCommand(context.Background(), "light", "turn_on", "light.bed_light", nil)
	secondID := h.lastCommand["id"]

	require.NotEqual(t, firstID, secondID)
}

func TestWSConnectorInvalidInputType(t *testing.T) {
	c := buildWSConnector(t, "http://example.com")
	_, err := c.Connect(context.Background(), "not-a-map")
	require.Error(t, err)
}

func TestWSConnectorTickAndStop(t *testing.T) {
	c := buildWSConnector(t, "http://example.com")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.Tick(ctx)

	c.Stop()
}
