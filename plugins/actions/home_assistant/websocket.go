package home_assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
)

const (
	defaultWSTimeout     = 10 * time.Second
	defaultWSBrightness  = 255
	defaultWSTemperature = 20.0
)

// WSConfig is the decoded plugin configuration for the WebSocket connector.
type WSConfig struct {
	// BaseURL is the root URL of the Home Assistant instance
	// (e.g. "http://homeassistant.local:8123"). Trailing slashes are ignored.
	BaseURL string `json:"base_url"`

	// Token is the Home Assistant long-lived access token.
	Token string `json:"token"`

	// Timeout is the connection/command timeout in seconds. 0 means use the default.
	Timeout float64 `json:"timeout"`
}

// WSConnector implements actions.Connector using the Home Assistant
// WebSocket API. Each Connect call opens a fresh connection, authenticates,
// sends one call_service command, and closes the connection.
type WSConnector struct {
	log     *zap.Logger
	wsURL   string
	token   string
	timeout time.Duration
	msgID   int
}

func init() {
	actions.Register("home_assistant/websocket", NewWSConnector)
}

// buildWSURL converts an http(s) base URL to a ws(s) Home Assistant
// websocket endpoint.
func buildWSURL(baseURL string) string {
	base := strings.TrimRight(baseURL, "/")
	switch {
	case strings.HasPrefix(base, "https://"):
		return "wss://" + strings.TrimPrefix(base, "https://") + "/api/websocket"
	case strings.HasPrefix(base, "http://"):
		return "ws://" + strings.TrimPrefix(base, "http://") + "/api/websocket"
	default:
		return base + "/api/websocket"
	}
}

// NewWSConnector builds the connector from a decoded config map.
func NewWSConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg WSConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("home_assistant/websocket")

	if cfg.BaseURL == "" {
		log.Warn("base_url not provided")
	}
	if cfg.Token == "" {
		log.Warn("token not provided")
	}

	timeout := defaultWSTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout * float64(time.Second))
	}

	return &WSConnector{
		log:     log,
		wsURL:   buildWSURL(cfg.BaseURL),
		token:   cfg.Token,
		timeout: timeout,
		msgID:   1,
	}, nil
}

// sendCommand opens a WebSocket connection, authenticates, sends a single
// call_service command, and reports whether it succeeded.
func (c *WSConnector) sendCommand(ctx context.Context, domain, service, entityID string, serviceData map[string]any) bool {
	if c.wsURL == "" || c.token == "" {
		return false
	}

	dialer := websocket.Dialer{HandshakeTimeout: c.timeout}
	conn, _, err := dialer.DialContext(ctx, c.wsURL, nil)
	if err != nil {
		c.log.Error("connection timed out or failed", zap.Error(err))
		return false
	}
	defer func() { _ = conn.Close() }()

	_ = conn.SetReadDeadline(time.Now().Add(c.timeout))

	// Step 1: expect auth_required
	var authRequired map[string]any
	if err := conn.ReadJSON(&authRequired); err != nil {
		c.log.Error("WebSocket error", zap.Error(err))
		return false
	}
	if authRequired["type"] != "auth_required" {
		c.log.Error("expected auth_required", zap.Any("got", authRequired["type"]))
		return false
	}

	// Step 2: send auth
	if err := conn.WriteJSON(map[string]any{
		"type":         "auth",
		"access_token": c.token,
	}); err != nil {
		c.log.Error("WebSocket error", zap.Error(err))
		return false
	}

	var authResult map[string]any
	if err := conn.ReadJSON(&authResult); err != nil {
		c.log.Error("WebSocket error", zap.Error(err))
		return false
	}
	if authResult["type"] != "auth_ok" {
		c.log.Error("authentication failed")
		return false
	}

	// Step 3: send call_service command
	if serviceData == nil {
		serviceData = map[string]any{}
	}
	msgID := c.msgID
	c.msgID++

	command := map[string]any{
		"id":      msgID,
		"type":    "call_service",
		"domain":  domain,
		"service": service,
		"target": map[string]any{
			"entity_id": entityID,
		},
		"service_data": serviceData,
	}
	if err := conn.WriteJSON(command); err != nil {
		c.log.Error("WebSocket error", zap.Error(err))
		return false
	}

	var result map[string]any
	if err := conn.ReadJSON(&result); err != nil {
		c.log.Error("WebSocket error", zap.Error(err))
		return false
	}

	if success, _ := result["success"].(bool); success {
		return true
	}

	errInfo, _ := result["error"].(map[string]any)
	c.log.Error("command failed",
		zap.Any("code", errInfo["code"]),
		zap.Any("message", errInfo["message"]),
	)
	return false
}

// Connect resolves the requested device action and sends the corresponding
// Home Assistant WebSocket command.
func (c *WSConnector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("home_assistant/websocket: unexpected input type %T", input)
	}

	entityID, _ := args["entity_id"].(string)
	deviceType, _ := args["device_type"].(string)
	action, _ := args["action"].(string)
	brightness, _ := args["brightness"].(string)
	temperature, _ := args["temperature"].(string)
	color, _ := args["color"].(string)

	if entityID == "" {
		return nil, nil
	}

	switch deviceType {
	case "light":
		c.connectLight(ctx, entityID, action, brightness, color)
	case "switch":
		c.connectSwitch(ctx, entityID, action)
	case "climate":
		c.connectClimate(ctx, entityID, action, temperature)
	default:
		c.log.Warn("device_type not supported", zap.String("device_type", deviceType))
	}

	return nil, nil
}

func (c *WSConnector) connectLight(ctx context.Context, entityID, action, brightness, color string) {
	switch action {
	case "turn_on":
		c.sendCommand(ctx, "light", "turn_on", entityID, nil)
	case "turn_off":
		c.sendCommand(ctx, "light", "turn_off", entityID, nil)
	case "set_brightness":
		v := parseBrightness(brightness)
		c.sendCommand(ctx, "light", "turn_on", entityID, map[string]any{"brightness": v})
	case "set_color":
		hs := resolveHSColor(color)
		c.sendCommand(ctx, "light", "turn_on", entityID, map[string]any{"hs_color": []int{hs[0], hs[1]}})
	default:
		c.log.Warn("action not supported for light", zap.String("action", action))
	}
}

func (c *WSConnector) connectSwitch(ctx context.Context, entityID, action string) {
	switch action {
	case "turn_on":
		c.sendCommand(ctx, "switch", "turn_on", entityID, nil)
	case "turn_off":
		c.sendCommand(ctx, "switch", "turn_off", entityID, nil)
	default:
		c.log.Warn("action not supported for switch", zap.String("action", action))
	}
}

func (c *WSConnector) connectClimate(ctx context.Context, entityID, action, temperature string) {
	switch action {
	case "turn_on":
		c.sendCommand(ctx, "climate", "turn_on", entityID, nil)
	case "turn_off":
		c.sendCommand(ctx, "climate", "turn_off", entityID, nil)
	case "set_temperature":
		t := parseTemperature(temperature, defaultWSTemperature)
		c.sendCommand(ctx, "climate", "set_temperature", entityID, map[string]any{"temperature": t})
	default:
		c.log.Warn("action not supported for climate", zap.String("action", action))
	}
}

// Tick blocks until ctx is cancelled; this connector is event-driven.
func (c *WSConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since each Connect call manages its own connection lifecycle.
func (c *WSConnector) Stop() {}
