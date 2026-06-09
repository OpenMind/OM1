package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("LocationsInput", NewLocations)
}

const (
	locationsDescriptor  = "These are the saved locations you can navigate to."
	locationsPollPeriod  = 500 * time.Millisecond
	locationsMaxMessages = 10

	simLocationsURL   = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/list"
	localLocationsURL = "http://localhost:5000/maps/locations/list"
)

type LocationsConfig struct {
	BaseURL         string `json:"base_url"`
	APIKey          string `json:"api_key"`
	UseSim          bool   `json:"use_sim"`
	Timeout         int    `json:"timeout"`
	RefreshInterval int    `json:"refresh_interval"`
}

type LocationsSensor struct {
	log      *zap.Logger
	provider *providers.LocationsProvider

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewLocations constructs a LocationsSensor with the given configuration.
func NewLocations(configMap map[string]any) (inputs.Sensor, error) {
	var cfg LocationsConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	if cfg.BaseURL == "" {
		if cfg.UseSim {
			cfg.BaseURL = simLocationsURL
		} else {
			cfg.BaseURL = localLocationsURL
		}
	}

	timeout := time.Duration(cfg.Timeout) * time.Second
	refresh := time.Duration(cfg.RefreshInterval) * time.Second

	log := logger.Get().Named("LocationsInput")
	log.Info("initializing", zap.String("base_url", cfg.BaseURL))

	provider := providers.NewLocationsProvider(cfg.BaseURL, cfg.APIKey, timeout, refresh)
	provider.Start()

	return &LocationsSensor{
		log:      log,
		provider: provider,
	}, nil
}

// Listen starts a goroutine that polls the provider at a regular interval and sends formatted snapshots to the output channel.
func (s *LocationsSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(locationsPollPeriod)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			raw, err := s.Poll(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
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

// Poll returns the latest set of saved locations formatted as one line each.
func (s *LocationsSensor) Poll(_ context.Context) (any, error) {
	locations := s.provider.AllLocations()

	lines := make([]string, 0, len(locations))
	for key, loc := range locations {
		label := loc.Name
		if label == "" {
			label = key
		}
		pos := loc.Pose.Position
		if pos.X != 0 || pos.Y != 0 {
			lines = append(lines, fmt.Sprintf("%s (x:%.2f y:%.2f)", label, pos.X, pos.Y))
		} else {
			lines = append(lines, label)
		}
	}
	sort.Strings(lines)

	s.log.Debug("formatted locations", zap.Int("count", len(lines)))
	return strings.Join(lines, "\n"), nil
}

// RawToText appends a new snapshot to the message buffer, returning it as an inputs.Message for downstream processing.
func (s *LocationsSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > locationsMaxMessages {
		s.messages = s.messages[len(s.messages)-locationsMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the most recent snapshot in the buffer.
func (s *LocationsSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", locationsDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput("LocationsInput", latest.Message, ts)
	s.messages = nil

	return result
}

func (s *LocationsSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
	if s.provider != nil {
		s.provider.Stop()
	}
}
