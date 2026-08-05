package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
	"github.com/openmind/om1/plugins/inputs/air_quality/connector"
	"github.com/openmind/om1/plugins/inputs/air_quality/connector/aqicn"
	"github.com/openmind/om1/plugins/inputs/air_quality/connector/bme680"
	"github.com/openmind/om1/plugins/inputs/air_quality/connector/pms5003"
)

func init() {
	inputs.Register("AirQuality", NewAirQuality)
}

const (
	airQualityDescriptor  = "Air Quality"
	airQualityIOKey       = "AirQuality"
	airQualityMaxMessages = 10

	airQualityDefaultConnector           = "aqicn"
	airQualityDefaultPollIntervalSec     = 300.0
	airQualityDefaultAQIWarningThreshold = 100
	airQualityDefaultAQIDangerThreshold  = 150
)

// AirQualityConfig mirrors the Python AirQualityConfig(SensorConfig).
// ConnectorConfig is passed through as-is (json.RawMessage) to whichever
// connector is selected, mirroring Python's `connector_config: dict`.
type AirQualityConfig struct {
	Connector           string          `json:"connector"`
	ConnectorConfig     json.RawMessage `json:"connector_config"`
	PollIntervalSec     float64         `json:"poll_interval"`
	AQIWarningThreshold int             `json:"aqi_warning_threshold"`
	AQIDangerThreshold  int             `json:"aqi_danger_threshold"`
}

type AirQualitySensor struct {
	cfg    AirQualityConfig
	log    *zap.Logger
	conn   connector.AirQualityConnector
	period time.Duration

	mu           sync.Mutex
	messages     []inputs.Message
	stopped      bool
	lastPollTime time.Time
}

// NewAirQuality builds an AirQualitySensor from the given config map,
// mirroring Python's AirQualityInput.__init__ (connector selection + defaults).
func NewAirQuality(configMap map[string]any) (inputs.Sensor, error) {
	var cfg AirQualityConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.Connector == "" {
		cfg.Connector = airQualityDefaultConnector
	}
	if cfg.PollIntervalSec <= 0 {
		cfg.PollIntervalSec = airQualityDefaultPollIntervalSec
	}
	if cfg.AQIWarningThreshold == 0 {
		cfg.AQIWarningThreshold = airQualityDefaultAQIWarningThreshold
	}
	if cfg.AQIDangerThreshold == 0 {
		cfg.AQIDangerThreshold = airQualityDefaultAQIDangerThreshold
	}

	log := logger.Get().Named("AirQuality")

	var conn connector.AirQualityConnector
	switch cfg.Connector {
	case "aqicn":
		var c aqicn.Config
		if len(cfg.ConnectorConfig) > 0 {
			_ = json.Unmarshal(cfg.ConnectorConfig, &c)
		}
		conn = aqicn.New(c, nil)
	case "pms5003":
		var c pms5003.Config
		if len(cfg.ConnectorConfig) > 0 {
			_ = json.Unmarshal(cfg.ConnectorConfig, &c)
		}
		conn = pms5003.New(c, nil)
	case "bme680":
		var c bme680.Config
		if len(cfg.ConnectorConfig) > 0 {
			_ = json.Unmarshal(cfg.ConnectorConfig, &c)
		}
		conn = bme680.New(c, nil)
	default:
		return nil, fmt.Errorf(
			"AirQualityInput: unknown connector '%s'. Available: [aqicn pms5003 bme680]",
			cfg.Connector,
		)
	}

	log.Info("initializing",
		zap.String("connector", cfg.Connector),
		zap.Float64("poll_interval", cfg.PollIntervalSec),
		zap.Int("aqi_warning_threshold", cfg.AQIWarningThreshold),
		zap.Int("aqi_danger_threshold", cfg.AQIDangerThreshold),
	)

	return &AirQualitySensor{
		cfg:    cfg,
		log:    log,
		conn:   conn,
		period: time.Duration(cfg.PollIntervalSec * float64(time.Second)),
	}, nil
}

// Listen polls the connector on a ticker and pushes formatted messages.
// Mirrors Python's `_poll()` loop being driven by the outer runtime, adapted
// to the OM1 Go Listen/channel pattern used by FacePresenceSensor.
func (s *AirQualitySensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(s.period)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			data, err := s.pollOnce(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Warn("poll failed", zap.Error(err))
				util.Sleep(ctx, 2*time.Second)
				continue
			}
			if data == nil {
				continue
			}

			msg := s.rawToMessage(data)
			if msg == nil {
				continue
			}

			s.mu.Lock()
			s.messages = append(s.messages, *msg)
			if len(s.messages) > airQualityMaxMessages {
				s.messages = s.messages[len(s.messages)-airQualityMaxMessages:]
			}
			s.mu.Unlock()

			providers.IO().AddInput(airQualityIOKey, msg.Message, time.Now())
		}
	}()
	return out, nil
}

// Poll performs one connect->read->disconnect cycle and returns the raw data.
// Mirrors Python's `_poll()`.
func (s *AirQualitySensor) Poll(ctx context.Context) (any, error) {
	return s.pollOnce(ctx)
}

// pollOnce mirrors Python's `_poll()`: connect, read, disconnect.
func (s *AirQualitySensor) pollOnce(ctx context.Context) (*connector.AirQualityData, error) {
	now := time.Now()
	if now.Sub(s.lastPollTime).Seconds() < s.cfg.PollIntervalSec {
		return nil, nil
	}
	s.lastPollTime = now

	ok, err := s.conn.Connect(ctx)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}

	data, err := s.conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	if derr := s.conn.Disconnect(ctx); derr != nil {
		s.log.Warn("disconnect error", zap.Error(derr))
	}

	return data, nil
}

// RawToText implements inputs.Sensor. Converts raw AirQualityData into a
// Message and appends it to the buffer. Mirrors Python's `raw_to_text`.
func (s *AirQualitySensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	data, ok := raw.(*connector.AirQualityData)
	if !ok || data == nil {
		return nil, nil
	}

	msg := s.rawToMessage(data)
	if msg == nil {
		return nil, nil
	}

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > airQualityMaxMessages {
		s.messages = s.messages[len(s.messages)-airQualityMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// rawToMessage converts AirQualityData into a human-readable message for the LLM.
// Mirrors Python's `_raw_to_text(raw_input)`.
func (s *AirQualitySensor) rawToMessage(raw *connector.AirQualityData) *inputs.Message {
	if raw == nil {
		return nil
	}

	var parts []string

	if raw.AQI != nil {
		label, _ := connector.GetAQILevel(*raw.AQI)
		parts = append(parts, fmt.Sprintf("Air Quality in %s: %s (AQI: %d)", raw.Location, label, *raw.AQI))
	} else {
		parts = append(parts, fmt.Sprintf("Air Quality in %s", raw.Location))
	}

	var pollutants []string
	if raw.PM25 != nil {
		pollutants = append(pollutants, fmt.Sprintf("PM2.5: %v µg/m³", *raw.PM25))
	}
	if raw.PM10 != nil {
		pollutants = append(pollutants, fmt.Sprintf("PM10: %v µg/m³", *raw.PM10))
	}
	if raw.CO != nil {
		pollutants = append(pollutants, fmt.Sprintf("CO: %v ppm", *raw.CO))
	}
	if raw.NO2 != nil {
		pollutants = append(pollutants, fmt.Sprintf("NO2: %v µg/m³", *raw.NO2))
	}
	if raw.SO2 != nil {
		pollutants = append(pollutants, fmt.Sprintf("SO2: %v µg/m³", *raw.SO2))
	}
	if raw.O3 != nil {
		pollutants = append(pollutants, fmt.Sprintf("O3: %v µg/m³", *raw.O3))
	}
	if len(pollutants) > 0 {
		parts = append(parts, strings.Join(pollutants, ", "))
	}

	var envData []string
	if raw.Temperature != nil {
		envData = append(envData, fmt.Sprintf("Temperature: %v°C", *raw.Temperature))
	}
	if raw.Humidity != nil {
		envData = append(envData, fmt.Sprintf("Humidity: %v%%", *raw.Humidity))
	}
	if len(envData) > 0 {
		parts = append(parts, strings.Join(envData, ", "))
	}

	if raw.AQI != nil {
		label, description := connector.GetAQILevel(*raw.AQI)
		if *raw.AQI >= s.cfg.AQIDangerThreshold {
			parts = append(parts, fmt.Sprintf("DANGER: Air quality is %s — %s", label, description))
		} else if *raw.AQI >= s.cfg.AQIWarningThreshold {
			parts = append(parts, fmt.Sprintf("WARNING: Air quality is %s — %s", label, description))
		}
	}

	text := strings.Join(parts, ". ") + "."
	return inputs.NewMessage(text)
}

// FormattedLatestBuffer returns the latest formatted air quality message,
// wrapped in the INPUT/START/END block, mirroring Python's
// `formatted_latest_buffer()`.
func (s *AirQualitySensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\nINPUT: %s\n// START\n%s\n// END\n", airQualityDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(airQualityIOKey, latest.Message, ts)
	s.messages = nil

	return result
}

func (s *AirQualitySensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
