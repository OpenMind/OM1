// Package robot_action provides an action plugin that triggers robot motions by
// POSTing to a configurable HTTP endpoint on the robot's local API server.
//
// Each supported LLM action name is mapped to:
//   - an endpoint path appended to base_url, and
//   - an optional JSON request body.
//
// Built-in defaults (override per-action via the plugin's "actions" config map):
//
//	wave -> POST {base_url}/motion  body: {"motion": "wave"}
//	shakehand -> POST {base_url}/motion  body: {"motion": "handshake"}
//
// To support a new robot motion:
//  1. Add the action name to the RobotAction enum's EnumValues() list.
//  2. Either add an entry to defaultActions or supply one in the plugin config
//     under "actions" (e.g. {"sit": {"path": "/sit"}}).
package robot_action

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/logger"
)

const (
	defaultBaseURL = "http://192.168.10.102:8080"
	defaultTimeout = 30 * time.Second
)

// RobotAction is the LLM-facing enum for the robot motion to trigger.
//
// Extend EnumValues() with additional names as new endpoints are added on the
// robot's HTTP API. The enum value is appended directly to base_url, so it must
// be a URL-safe path segment that matches the server-side route.
type RobotAction string

// EnumValues lists the robot motions the LLM may request.
func (RobotAction) EnumValues() []string {
	return []string{
		"wave",
		"shakehand",
	}
}

// RobotActionInput is the input the LLM provides when calling this action.
type RobotActionInput struct {
	Action RobotAction `json:"action" description:"The robot motion to trigger. Use 'wave' when the user asks the robot to wave, and 'shakehand' when the user asks to shake hands. The value selects a configured endpoint + body on the robot's HTTP API."`
}

// ActionMapping describes how a single LLM-facing action maps to an HTTP call.
type ActionMapping struct {
	// Path is the endpoint path relative to base_url (e.g. "/handshake").
	// If empty, "/{action_name}" is used.
	Path string `json:"path"`

	// Body is an optional JSON body sent with the POST request. When nil,
	// the request is sent without a body.
	Body map[string]any `json:"body"`
}

// defaultActions provides the built-in action -> endpoint mapping. Users may
// override or extend it via the plugin's "actions" config map.
var defaultActions = map[string]ActionMapping{
	"wave": {
		Path: "/motion",
		Body: map[string]any{"motion": "wave"},
	},
	"shakehand": {
		Path: "/motion",
		Body: map[string]any{"motion": "handshake"},
	},
}

func init() {
	actions.RegisterInterface(
		"robot_action",
		"Action interface that triggers a physical robot motion by POSTing to an "+
			"HTTP endpoint on the robot's local API server. Use this when the user "+
			"asks the robot to perform one of the supported motions (e.g. \"wave "+
			"hands\" -> action=\"wave\", \"shake hands\" -> action=\"shakehand\"). "+
			"The 'action' field is appended to the configured base_url when no custom "+
			"mapping is configured.",
		RobotActionInput{},
	)
	actions.Register("robot_action/http", NewHTTPConnector)
}

// Config is the decoded plugin configuration.
type Config struct {
	// BaseURL is the root URL of the robot HTTP API
	// (e.g. "http://192.168.10.102:8080"). Trailing slashes are ignored.
	BaseURL string `json:"base_url"`

	// Timeout is the per-request timeout in seconds. 0 means use the default.
	Timeout int `json:"timeout"`

	// Actions overrides or extends the built-in action -> endpoint mapping.
	// Keys are LLM-facing action names and must also appear in
	// RobotAction.EnumValues() for the LLM to be able to call them.
	Actions map[string]ActionMapping `json:"actions"`
}

// Connector implements actions.Connector by POSTing to {baseURL}/{path} for
// each LLM-requested motion, where path and body come from the action mapping.
type Connector struct {
	log     *zap.Logger
	baseURL string
	timeout time.Duration
	actions map[string]ActionMapping
}

// NewHTTPConnector builds the connector from a decoded config map.
func NewHTTPConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	cfg.BaseURL = strings.TrimRight(cfg.BaseURL, "/")

	timeout := defaultTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout) * time.Second
	}

	mappings := make(map[string]ActionMapping, len(defaultActions)+len(cfg.Actions))
	for name, mapping := range defaultActions {
		mappings[name] = mapping
	}
	for name, mapping := range cfg.Actions {
		mappings[name] = mapping
	}

	log := logger.Get().Named("robot_action/http")
	log.Info("connector initialized",
		zap.String("base_url", cfg.BaseURL),
		zap.Duration("timeout", timeout),
		zap.Int("action_count", len(mappings)),
	)

	return &Connector{
		log:     log,
		baseURL: cfg.BaseURL,
		timeout: timeout,
		actions: mappings,
	}, nil
}

// Connect resolves the action mapping for the requested motion and issues
// POST {baseURL}{mapping.path} with mapping.body (if any).
func (c *Connector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("robot_action/http: unexpected input type %T", input)
	}

	action, _ := args["action"].(string)
	action = strings.TrimSpace(action)
	if action == "" {
		return nil, nil
	}

	mapping, known := c.actions[action]
	if !known {
		c.log.Warn("no mapping for action; falling back to /{action}",
			zap.String("action", action))
		mapping = ActionMapping{Path: "/" + action}
	}

	path := mapping.Path
	if path == "" {
		path = "/" + action
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	url := c.baseURL + path

	var bodyReader io.Reader
	var bodyBytes []byte
	if mapping.Body != nil {
		var err error
		bodyBytes, err = json.Marshal(mapping.Body)
		if err != nil {
			return nil, fmt.Errorf("robot_action/http: marshal body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("robot_action/http: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	c.log.Info("triggering robot action",
		zap.String("action", action),
		zap.String("url", url),
		zap.ByteString("body", bodyBytes),
	)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		c.log.Error("API request failed",
			zap.String("action", action),
			zap.String("url", url),
			zap.Error(err),
		)
		return nil, nil
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		c.log.Info("robot action completed",
			zap.String("action", action),
			zap.Int("status", resp.StatusCode),
			zap.ByteString("response", respBody),
		)
	} else {
		c.log.Error("API returned error",
			zap.String("action", action),
			zap.Int("status", resp.StatusCode),
			zap.ByteString("response", respBody),
		)
	}

	return nil, nil
}

// Tick blocks until ctx is cancelled; this connector is event-driven.
func (c *Connector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the shared HTTP client manages its own resources.
func (c *Connector) Stop() {}
