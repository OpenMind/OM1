package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("SmokeGasDetector", NewSmokeGasDetector)
}

const (
	smokeGasDescriptor  = "Smoke and Gas Detector"
	smokeGasIOKey       = "SmokeGasDetector"
	smokeGasMaxMessages = 1

	smokeWarningThresholdDefault = 300.0
	smokeDangerThresholdDefault  = 600.0
	gasWarningThresholdDefault   = 300.0
	gasDangerThresholdDefault    = 600.0
	cooldownDefaultSec           = 5.0
	smokeGasPollIntervalSec      = 0.5
)

type SmokeGasDetectorConfig struct {
	CooldownSec           float64 `json:"cooldown"`
	SmokeWarningThreshold float64 `json:"smoke_warning_threshold"`
	SmokeDangerThreshold  float64 `json:"smoke_danger_threshold"`
	GasWarningThreshold   float64 `json:"gas_warning_threshold"`
	GasDangerThreshold    float64 `json:"gas_danger_threshold"`
	MockScenario          string  `json:"mock_scenario"`
}

type SmokeGasReading struct {
	SmokePPM float64
	GasPPM   float64
}

type mockSmokeConnector struct {
	scenario string
	rng      *rand.Rand
}

func newMockSmokeConnector(scenario string) *mockSmokeConnector {
	return &mockSmokeConnector{
		scenario: scenario,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (m *mockSmokeConnector) Read() *SmokeGasReading {
	var smoke, gas float64
	switch m.scenario {
	case "warning":
		smoke = 350 + (m.rng.Float64()*40 - 20)
		gas = 320 + (m.rng.Float64()*40 - 20)
	case "danger":
		smoke = 750 + (m.rng.Float64()*60 - 30)
		gas = 700 + (m.rng.Float64()*60 - 30)
	default:
		smoke = 50 + (m.rng.Float64()*20 - 10)
		gas = 40 + (m.rng.Float64()*20 - 10)
	}
	return &SmokeGasReading{SmokePPM: smoke, GasPPM: gas}
}

type SmokeGasDetectorSensor struct {
	cfg       SmokeGasDetectorConfig
	log       *zap.Logger
	connector *mockSmokeConnector

	mu            sync.Mutex
	messages      []inputs.Message
	lastAlertTime time.Time
	stopped       bool
}

func NewSmokeGasDetector(configMap map[string]any) (inputs.Sensor, error) {
	var cfg SmokeGasDetectorConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	if cfg.CooldownSec <= 0 {
		cfg.CooldownSec = cooldownDefaultSec
	}
	if cfg.SmokeWarningThreshold <= 0 {
		cfg.SmokeWarningThreshold = smokeWarningThresholdDefault
	}
	if cfg.SmokeDangerThreshold <= 0 {
		cfg.SmokeDangerThreshold = smokeDangerThresholdDefault
	}
	if cfg.GasWarningThreshold <= 0 {
		cfg.GasWarningThreshold = gasWarningThresholdDefault
	}
	if cfg.GasDangerThreshold <= 0 {
		cfg.GasDangerThreshold = gasDangerThresholdDefault
	}
	if cfg.MockScenario == "" {
		cfg.MockScenario = "normal"
	}

	log := logger.Get().Named("SmokeGasDetector")
	log.Info("initialized", zap.String("mock_scenario", cfg.MockScenario), zap.Float64("cooldown_sec", cfg.CooldownSec))

	return &SmokeGasDetectorSensor{
		cfg:       cfg,
		log:       log,
		connector: newMockSmokeConnector(cfg.MockScenario),
	}, nil
}

func (s *SmokeGasDetectorSensor) classify(r *SmokeGasReading) string {
	if r.SmokePPM >= s.cfg.SmokeDangerThreshold || r.GasPPM >= s.cfg.GasDangerThreshold {
		return "danger"
	}
	if r.SmokePPM >= s.cfg.SmokeWarningThreshold || r.GasPPM >= s.cfg.GasWarningThreshold {
		return "warning"
	}
	return "normal"
}

func (s *SmokeGasDetectorSensor) readingToText(r *SmokeGasReading) string {
	level := s.classify(r)
	now := time.Now()
	cooldown := time.Duration(s.cfg.CooldownSec * float64(time.Second))

	switch level {
	case "danger":
		if now.Sub(s.lastAlertTime) < cooldown {
			return ""
		}
		s.lastAlertTime = now
		return fmt.Sprintf(
			"SMOKE ALERT: Critical smoke/gas level detected. Smoke: %.0f ppm, Gas: %.0f ppm. Immediate evacuation recommended. Possible fire or gas leak.",
			r.SmokePPM, r.GasPPM,
		)
	case "warning":
		if now.Sub(s.lastAlertTime) < cooldown {
			return ""
		}
		s.lastAlertTime = now
		return fmt.Sprintf(
			"SMOKE WARNING: Elevated smoke/gas detected. Smoke: %.0f ppm, Gas: %.0f ppm. Possible fire risk. Inspect area immediately.",
			r.SmokePPM, r.GasPPM,
		)
	default:
		return fmt.Sprintf("Smoke/gas detector: Air quality normal. Smoke: %.0f ppm, Gas: %.0f ppm.", r.SmokePPM, r.GasPPM)
	}
}

func (s *SmokeGasDetectorSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(time.Duration(smokeGasPollIntervalSec * float64(time.Second)))
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			text := s.readingToText(s.connector.Read())
			if text == "" {
				continue
			}

			msg := inputs.NewMessage(text)
			s.mu.Lock()
			s.messages = append(s.messages, *msg)
			if len(s.messages) > smokeGasMaxMessages {
				s.messages = s.messages[len(s.messages)-smokeGasMaxMessages:]
			}
			s.mu.Unlock()
		}
	}()
	return out, nil
}

func (s *SmokeGasDetectorSensor) Poll(_ context.Context) (any, error) {
	return s.readingToText(s.connector.Read()), nil
}

func (s *SmokeGasDetectorSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}
	msg := inputs.NewMessage(text)
	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	s.mu.Unlock()
	return msg, nil
}

func (s *SmokeGasDetectorSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\nINPUT: %s\n// START\n%s\n// END\n", smokeGasDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(smokeGasIOKey, latest.Message, ts)
	s.messages = nil

	return result
}

func (s *SmokeGasDetectorSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
