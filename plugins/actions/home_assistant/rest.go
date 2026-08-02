package home_assistant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/logger"
)

const (
	defaultRESTTimeout     = 10 * time.Second
	defaultRESTBrightness  = 255
	defaultRESTTemperature = 22.0
)

// RESTConfig is the decoded plugin configuration for the REST connector.
type RESTConfig struct {
	// BaseURL is the root URL of the Home Assistant instance
	// (e.g. "http://homeassistant.local:8123"). Trailing slashes are ignored.
	BaseURL string `json:"base_url"`

	// Token is the Home Assistant long-lived access token.
	Token string `json:"token"`

	// Timeout is the per-request timeout in seconds. 0 means use the default.
	Timeout float64 `json:"timeout"`
}

// RESTConnector implements actions.Connector by calling the Home Assistant
// REST API (POST /api/services/{domain}/{service}) for each LLM-requested
// device action.
type RESTConnector struct {
	log     *zap.Logger
	baseURL string
	token   string
	timeout time.Duration
}

func init() {
	actions.Register("home_assistant/rest", NewRESTConnector)
}

// NewRESTConnector builds the connector from a decoded config map.
func NewRESTConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg RESTConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("home_assistant/rest")

	if cfg.BaseURL == "" {
		log.Warn("base_url not provided in configuration")
	}
	if cfg.Token == "" {
		log.Warn("token not provided in configuration")
	}

	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")

	timeout := defaultRESTTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout * float64(time.Second))
	}

	return &RESTConnector{
		log:     log,
		baseURL: cfg.BaseURL,
		token:   cfg.Token,
		timeout: timeout,
	}, nil
}

// parseBrightness converts a brightness string to an int (0-255), clamping
// out-of-range input and falling back to defaultRESTBrightness on
// empty/invalid input. Brightness is a direct 0-255 value, not a percentage.
func parseBrightness(s string) int {
	v := defaultRESTBrightness
	if s != "" {
		if parsed, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			v = parsed
		}
	}
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return v
}

// parseTemperature converts a temperature string to a float64, falling back
// to the given default on empty/invalid input.
func parseTemperature(s string, fallback float64) float64 {
	if s == "" {
		return fallback
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return fallback
	}
	return v
}

// resolveHSColor looks up the HS (hue, saturation) value for a color name,
// falling back to white on unknown/empty input.
func resolveHSColor(color string) [2]int {
	hs, ok := COLOR_MAP[strings.ToLower(strings.TrimSpace(color))]
	if !ok {
		return COLOR_MAP["white"]
	}
	return hs
}

// callService issues POST {baseURL}/api/services/{domain}/{service} with the
// given JSON body.
func (c *RESTConnector) callService(ctx context.Context, domain, service string, body map[string]any) error {
	if c.baseURL == "" || c.token == "" {
		c.log.Error("base_url or token not set")
		return nil
	}

	url := fmt.Sprintf("%s/api/services/%s/%s", c.baseURL, domain, service)

	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("home_assistant/rest: marshal body: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("home_assistant/rest: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.token)

	c.log.Info("POST request",
		zap.String("url", url),
		zap.ByteString("payload", payload),
	)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		c.log.Error("network error", zap.String("url", url), zap.Error(err))
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		c.log.Error("error response",
			zap.Int("status", resp.StatusCode),
			zap.ByteString("response", respBody),
		)
		return fmt.Errorf("home_assistant/rest: status %d", resp.StatusCode)
	}

	c.log.Info("success",
		zap.Int("status", resp.StatusCode),
	)
	return nil
}

// Connect resolves the requested device action and calls the corresponding
// Home Assistant service.
func (c *RESTConnector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("home_assistant/rest: unexpected input type %T", input)
	}

	entityID, _ := args["entity_id"].(string)
	deviceType, _ := args["device_type"].(string)
	action, _ := args["action"].(string)
	brightness, _ := args["brightness"].(string)
	temperature, _ := args["temperature"].(string)
	color, _ := args["color"].(string)

	if entityID == "" {
		c.log.Warn("entity_id is empty, skipping")
		return nil, nil
	}

	c.log.Info("dispatching action",
		zap.String("action", action),
		zap.String("entity_id", entityID),
		zap.String("device_type", deviceType),
	)

	switch deviceType {
	case "light":
		return nil, c.connectLight(ctx, entityID, action, brightness, color)
	case "switch":
		return nil, c.connectSwitch(ctx, entityID, action)
	case "climate":
		return nil, c.connectClimate(ctx, entityID, action, temperature)
	default:
		c.log.Warn("device_type not supported", zap.String("device_type", deviceType))
		return nil, nil
	}
}

func (c *RESTConnector) connectLight(ctx context.Context, entityID, action, brightness, color string) error {
	switch action {
	case "turn_on":
		return c.callService(ctx, "light", "turn_on", map[string]any{"entity_id": entityID})
	case "turn_off":
		return c.callService(ctx, "light", "turn_off", map[string]any{"entity_id": entityID})
	case "set_brightness":
		return c.callService(ctx, "light", "turn_on", map[string]any{
			"entity_id":  entityID,
			"brightness": parseBrightness(brightness),
		})
	case "set_color":
		hs := resolveHSColor(color)
		return c.callService(ctx, "light", "turn_on", map[string]any{
			"entity_id": entityID,
			"hs_color":  []int{hs[0], hs[1]},
		})
	default:
		c.log.Warn("action not supported for light", zap.String("action", action))
		return nil
	}
}

func (c *RESTConnector) connectSwitch(ctx context.Context, entityID, action string) error {
	switch action {
	case "turn_on":
		return c.callService(ctx, "switch", "turn_on", map[string]any{"entity_id": entityID})
	case "turn_off":
		return c.callService(ctx, "switch", "turn_off", map[string]any{"entity_id": entityID})
	default:
		c.log.Warn("action not supported for switch", zap.String("action", action))
		return nil
	}
}

func (c *RESTConnector) connectClimate(ctx context.Context, entityID, action, temperature string) error {
	switch action {
	case "turn_on":
		return c.callService(ctx, "climate", "turn_on", map[string]any{"entity_id": entityID})
	case "turn_off":
		return c.callService(ctx, "climate", "turn_off", map[string]any{"entity_id": entityID})
	case "set_temperature":
		return c.callService(ctx, "climate", "set_temperature", map[string]any{
			"entity_id":   entityID,
			"temperature": parseTemperature(temperature, defaultRESTTemperature),
		})
	default:
		c.log.Warn("action not supported for climate", zap.String("action", action))
		return nil
	}
}

// Tick blocks until ctx is cancelled; this connector is event-driven.
func (c *RESTConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the shared HTTP client manages its own resources.
func (c *RESTConnector) Stop() {}
