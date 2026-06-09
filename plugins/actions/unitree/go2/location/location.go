package location

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
)

const (
	simBaseURL   = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/add/slam"
	localBaseURL = "http://localhost:5000/maps/locations/add/slam"

	defaultTimeout = 5 * time.Second
	defaultMapName = "map"
)

type RememberLocationInput struct {
	Action      string `json:"action" description:"The location name to save (e.g. \"kitchen\", \"office\", \"living room\")"`
	Description string `json:"description" description:"Optional additional information about the location"`
}

func init() {
	actions.RegisterInterface(
		"unitree_go2_location",
		"Action interface to save/remember the robot's current location with a name. "+
			"The 'action' field should contain the location name. Extract the location name "+
			"from user commands like \"remember this as [name]\" or \"save this location as [name]\".",
		RememberLocationInput{},
	)
	actions.Register("unitree_go2_location/location", NewUnitreeGo2RememberLocation)
}

type Config struct {
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
	UseSim  bool   `json:"use_sim"`
	Timeout int    `json:"timeout"`
	MapName string `json:"map_name"`

	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
}

type Connector struct {
	log     *zap.Logger
	tts     *tts.ElevenLabsProvider
	baseURL string
	apiKey  string
	timeout time.Duration
	mapName string
}

// parseConfig decodes the raw config map and applies defaults.
func parseConfig(configMap map[string]any) Config {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	if cfg.BaseURL == "" {
		if cfg.UseSim {
			cfg.BaseURL = simBaseURL
		} else {
			cfg.BaseURL = localBaseURL
		}
	}
	if cfg.MapName == "" {
		cfg.MapName = defaultMapName
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = tts.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = tts.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = tts.DefaultRate
	}

	return cfg
}

// NewUnitreeGo2RememberLocation builds the connector from a decoded config map.
func NewUnitreeGo2RememberLocation(configMap map[string]any) (actions.Connector, error) {
	cfg := parseConfig(configMap)

	timeout := defaultTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout) * time.Second
	}

	log := logger.Get().Named("unitree_go2_location/location")

	elevenlabs := tts.ElevenLabs(tts.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	return &Connector{
		log:     log,
		tts:     elevenlabs,
		baseURL: cfg.BaseURL,
		apiKey:  cfg.APIKey,
		timeout: timeout,
		mapName: cfg.MapName,
	}, nil
}

// Connect persists the remembered location and announces success via TTS.
func (c *Connector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unitree_go2_location/location: unexpected input type %T", input)
	}

	label, _ := args["action"].(string)
	if label == "" {
		return nil, nil
	}
	description, _ := args["description"].(string)

	if c.baseURL == "" {
		c.log.Error("missing base_url in config")
		return nil, nil
	}

	payload := map[string]any{
		"map_name":    c.mapName,
		"label":       label,
		"description": description,
	}
	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("unitree_go2_location/location: marshal payload: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("unitree_go2_location/location: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		c.log.Error("API request failed", zap.Error(err))
		return nil, nil
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		c.log.Info("stored location",
			zap.String("label", label),
			zap.Int("status", resp.StatusCode),
			zap.ByteString("response", respBody),
		)
		c.tts.AddText(fmt.Sprintf("Location %s remembered! Woof! Woof!", label))
	} else {
		c.log.Error("API returned error",
			zap.Int("status", resp.StatusCode),
			zap.ByteString("response", respBody),
		)
	}

	return nil, nil
}

// Tick is a no-op since this connector is event-driven.
func (c *Connector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the shared ElevenLabs provider manages its own resources.
func (c *Connector) Stop() {}
