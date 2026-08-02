// Package home_assistant provides an input sensor that polls Home Assistant
// entity states and reports changes to the LLM as they occur.
package home_assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

const (
	descriptorForLLM  = "Home Assistant Device States"
	defaultPollPeriod = 1 * time.Second
	defaultInterval   = 30.0
	fetchTimeout      = 10 * time.Second
)

func init() {
	inputs.Register("HomeAssistantInput", NewHomeAssistantInput)
}

// Config is the decoded plugin configuration for the Home Assistant state input.
type Config struct {
	// BaseURL is the root URL of the Home Assistant instance.
	BaseURL string `json:"base_url"`

	// Token is the Home Assistant long-lived access token.
	Token string `json:"token"`

	// EntityIDs is a comma-separated list of entity IDs to monitor.
	EntityIDs string `json:"entity_ids"`

	// PollInterval is the number of seconds between state polls. 0 means
	// use the default (30s).
	PollInterval float64 `json:"poll_interval"`
}

// StateSensor polls Home Assistant entity states and reports changes.
type StateSensor struct {
	log *zap.Logger

	baseURL      string
	token        string
	entityIDs    []string
	pollInterval time.Duration

	mu           sync.Mutex
	messages     []inputs.Message
	lastPollTime time.Time
	lastStates   map[string]string
	stopped      bool
}

// NewHomeAssistantInput constructs a StateSensor from a decoded config map.
func NewHomeAssistantInput(configMap map[string]any) (inputs.Sensor, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("HomeAssistantInput")

	if cfg.BaseURL == "" {
		log.Warn("base_url not provided")
	}
	if cfg.Token == "" {
		log.Warn("token not provided")
	}

	var entityIDs []string
	for _, e := range strings.Split(cfg.EntityIDs, ",") {
		e = strings.TrimSpace(e)
		if e != "" {
			entityIDs = append(entityIDs, e)
		}
	}
	if len(entityIDs) == 0 {
		log.Warn("no entity_ids configured")
	}

	interval := defaultInterval
	if cfg.PollInterval > 0 {
		interval = cfg.PollInterval
	}

	return &StateSensor{
		log:          log,
		baseURL:      strings.TrimRight(cfg.BaseURL, "/"),
		token:        cfg.Token,
		entityIDs:    entityIDs,
		pollInterval: time.Duration(interval * float64(time.Second)),
		lastStates:   map[string]string{},
	}, nil
}

// fetchState fetches the current state of a single entity from Home Assistant.
func (s *StateSensor) fetchState(ctx context.Context, entityID string) (map[string]any, error) {
	if s.baseURL == "" || s.token == "" {
		return nil, nil
	}

	url := fmt.Sprintf("%s/api/states/%s", s.baseURL, entityID)

	reqCtx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+s.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		s.log.Error("network error fetching entity", zap.String("entity_id", entityID), zap.Error(err))
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		s.log.Error("error fetching entity",
			zap.String("entity_id", entityID),
			zap.Int("status", resp.StatusCode),
		)
		return nil, fmt.Errorf("home_assistant input: status %d", resp.StatusCode)
	}

	var state map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&state); err != nil {
		return nil, err
	}
	return state, nil
}

// Poll fetches states for all configured entities, but only once per
// pollInterval; it returns nil in between, matching the throttled polling
// behavior of the original implementation.
func (s *StateSensor) Poll(ctx context.Context) (any, error) {
	s.mu.Lock()
	elapsed := time.Since(s.lastPollTime)
	shouldPoll := elapsed >= s.pollInterval
	if shouldPoll {
		s.lastPollTime = time.Now()
	}
	s.mu.Unlock()

	if !shouldPoll {
		return nil, nil
	}

	if len(s.entityIDs) == 0 {
		return nil, nil
	}

	var states []map[string]any
	for _, entityID := range s.entityIDs {
		state, err := s.fetchState(ctx, entityID)
		if err != nil || state == nil {
			continue
		}
		states = append(states, state)
	}

	if len(states) == 0 {
		return nil, nil
	}
	return states, nil
}

// Listen starts a goroutine that polls at defaultPollPeriod and forwards
// non-nil results to the output channel.
func (s *StateSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(defaultPollPeriod)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			raw, err := s.Poll(ctx)
			if err != nil || raw == nil {
				continue
			}

			select {
			case out <- raw:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// formatState formats a single entity state into human-readable text.
func formatState(state map[string]any) string {
	entityID, _ := state["entity_id"].(string)
	if entityID == "" {
		entityID = "unknown"
	}
	currentState, _ := state["state"].(string)
	if currentState == "" {
		currentState = "unknown"
	}
	attributes, _ := state["attributes"].(map[string]any)
	if attributes == nil {
		attributes = map[string]any{}
	}
	friendlyName, _ := attributes["friendly_name"].(string)
	if friendlyName == "" {
		friendlyName = entityID
	}

	parts := []string{fmt.Sprintf("%s (%s) is %s", friendlyName, entityID, currentState)}

	if brightness, ok := attributes["brightness"].(float64); ok {
		pct := int(brightness/255*100 + 0.5)
		parts = append(parts, fmt.Sprintf("brightness %d%%", pct))
	}

	if colorName, ok := attributes["color_name"].(string); ok {
		parts = append(parts, fmt.Sprintf("color %s", colorName))
	}

	if temperature, ok := attributes["temperature"]; ok {
		parts = append(parts, fmt.Sprintf("temperature %v°C", temperature))
	}

	if currentTemp, ok := attributes["current_temperature"]; ok {
		parts = append(parts, fmt.Sprintf("current temperature %v°C", currentTemp))
	}

	return strings.Join(parts, ", ")
}

// RawToText compares the incoming states against the last known states and
// builds a Message describing only the entities that changed.
func (s *StateSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	states, ok := raw.([]map[string]any)
	if !ok || len(states) == 0 {
		return nil, nil
	}

	s.mu.Lock()
	var changed []map[string]any
	for _, state := range states {
		entityID, _ := state["entity_id"].(string)
		currentState, _ := state["state"].(string)
		if s.lastStates[entityID] != currentState {
			changed = append(changed, state)
			s.lastStates[entityID] = currentState
		}
	}
	s.mu.Unlock()

	if len(changed) == 0 {
		return nil, nil
	}

	lines := make([]string, 0, len(changed))
	for _, state := range changed {
		lines = append(lines, formatState(state))
	}
	text := "Smart home device updates: " + strings.Join(lines, "; ")

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer formats and clears the latest buffered message.
func (s *StateSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\nINPUT: %s\n// START\n%s\n// END\n", descriptorForLLM, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(descriptorForLLM, latest.Message, ts)
	s.messages = nil

	return result
}

// Stop marks the sensor as stopped. Safe to call multiple times.
func (s *StateSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
