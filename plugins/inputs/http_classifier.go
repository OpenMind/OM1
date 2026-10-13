package inputs

// HTTPClassifierInput polls a local signal, sends it to an external HTTP classification service, and
// surfaces the service's response as an input.
//
// Config:
//
//	{ type: "HTTPClassifierInput", config: { endpoint_url: "http://127.0.0.1:5557", period_ms: 2000,
//	                                         source: "paths", min_confidence: 0.5 } }

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("HTTPClassifierInput", NewHTTPClassifier)
}

const (
	httpClassifierDescriptor     = "External classifier"
	httpClassifierDefaultURL     = "http://127.0.0.1:5557"
	httpClassifierDefaultPeriod  = 2000 * time.Millisecond
	httpClassifierDefaultTimeout = 1500 * time.Millisecond
	httpClassifierMaxMessages    = 10
)

type HTTPClassifierConfig struct {
	EndpointURL    string  `json:"endpoint_url"`
	PeriodMs       int     `json:"period_ms"`
	TimeoutMs      int     `json:"timeout_ms"`
	Source         string  `json:"source"`     // "paths" | "file"
	LabelFile      string  `json:"label_file"` // source "file"
	ReportNegative bool    `json:"report_negative"`
	MinConfidence  float64 `json:"min_confidence"`
}

type httpClassifierResponse struct {
	Seq        int     `json:"seq"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	LatencyMs  float64 `json:"latency_ms"`
	Error      string  `json:"error,omitempty"`
}

type HTTPClassifierSensor struct {
	cfg    HTTPClassifierConfig
	log    *zap.Logger
	client *http.Client
	paths  *providers.PathsProvider
	period time.Duration

	mu       sync.Mutex
	seq      int
	messages []inputs.Message
	stopped  bool
}

// NewHTTPClassifier constructs the sensor from its config map.
func NewHTTPClassifier(configMap map[string]any) (inputs.Sensor, error) {
	var cfg HTTPClassifierConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.EndpointURL == "" {
		cfg.EndpointURL = httpClassifierDefaultURL
	}
	cfg.EndpointURL = strings.TrimRight(cfg.EndpointURL, "/")
	if cfg.Source == "" {
		cfg.Source = "paths"
	}
	if cfg.Source != "paths" && cfg.Source != "file" {
		return nil, fmt.Errorf("http_classifier: source must be \"paths\" or \"file\", got %q", cfg.Source)
	}
	if cfg.Source == "file" && cfg.LabelFile == "" {
		return nil, fmt.Errorf("http_classifier: source \"file\" needs label_file")
	}
	period := httpClassifierDefaultPeriod
	if cfg.PeriodMs > 0 {
		period = time.Duration(cfg.PeriodMs) * time.Millisecond
	}
	timeout := httpClassifierDefaultTimeout
	if cfg.TimeoutMs > 0 {
		timeout = time.Duration(cfg.TimeoutMs) * time.Millisecond
	}

	log := logger.Get().Named("HTTPClassifier")
	log.Info("initializing", zap.String("endpoint_url", cfg.EndpointURL), zap.String("source", cfg.Source), zap.Duration("period", period))

	s := &HTTPClassifierSensor{
		cfg:    cfg,
		log:    log,
		client: &http.Client{Timeout: timeout},
		period: period,
	}
	if cfg.Source == "paths" {
		s.paths = providers.NewPathsProvider()
	}
	return s, nil
}

// label derives the current A/B label from the configured source.
func (s *HTTPClassifierSensor) label() (string, error) {
	switch s.cfg.Source {
	case "paths":
		m := s.paths.Movement()
		if m.Retreat || len(m.Advance) == 0 {
			return "A", nil
		}
		return "B", nil
	case "file":
		b, err := os.ReadFile(s.cfg.LabelFile)
		if err != nil {
			return "", err
		}
		t := strings.ToUpper(strings.TrimSpace(string(b)))
		if t == "" {
			return "B", nil
		}
		switch t[0] {
		case 'A':
			return "A", nil
		case 'B':
			return "B", nil
		}
		return "", fmt.Errorf("http_classifier: label file must start with A or B, got %q", t)
	}
	return "", fmt.Errorf("http_classifier: unknown source %q", s.cfg.Source)
}

// classify POSTs one label to the endpoint and returns the response.
func (s *HTTPClassifierSensor) classify(ctx context.Context, label string) (*httpClassifierResponse, error) {
	s.mu.Lock()
	s.seq++
	seq := s.seq
	s.mu.Unlock()

	body, _ := json.Marshal(map[string]any{"seq": seq, "symbol": label, "t_robot": float64(time.Now().UnixNano()) / 1e9})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.EndpointURL+"/classify", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var r httpClassifierResponse
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, fmt.Errorf("http_classifier: bad response json: %w", err)
	}
	if r.Error != "" {
		return &r, fmt.Errorf("http_classifier: endpoint error: %s", r.Error)
	}
	if resp.StatusCode != http.StatusOK {
		return &r, fmt.Errorf("http_classifier: endpoint HTTP %d", resp.StatusCode)
	}
	return &r, nil
}

// Listen polls at the configured period and yields each response until ctx is cancelled.
func (s *HTTPClassifierSensor) Listen(ctx context.Context) (<-chan any, error) {
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

			raw, err := s.Poll(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Warn("poll failed", zap.Error(err))
				continue
			}
			if raw == nil {
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

// Poll derives the current label, queries the endpoint, and returns the response.
func (s *HTTPClassifierSensor) Poll(ctx context.Context) (any, error) {
	label, err := s.label()
	if err != nil {
		return nil, err
	}
	return s.classify(ctx, label)
}

// RawToText turns a response into a message.
func (s *HTTPClassifierSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	r, ok := raw.(*httpClassifierResponse)
	if !ok || r == nil {
		return nil, nil
	}
	var text string
	switch r.Label {
	case "A":
		if r.Confidence < s.cfg.MinConfidence {
			return nil, nil
		}
		text = fmt.Sprintf("external classifier confirms change (%.2f, %.0f ms)", r.Confidence, r.LatencyMs)
	case "B":
		if !s.cfg.ReportNegative {
			return nil, nil
		}
		text = fmt.Sprintf("external classifier reports no change (%.2f, %.0f ms)", r.Confidence, r.LatencyMs)
	default:
		if !s.cfg.ReportNegative {
			return nil, nil
		}
		text = "external classifier returned no result"
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > httpClassifierMaxMessages {
		s.messages = s.messages[len(s.messages)-httpClassifierMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the most recent message and clears the buffer.
func (s *HTTPClassifierSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: %q\n", httpClassifierDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(httpClassifierDescriptor, latest.Message, ts)
	s.messages = nil

	return result
}

// TriggersTick wakes the cortex loop when a message is available.
func (s *HTTPClassifierSensor) TriggersTick() bool {
	return true
}

// Stop releases the paths provider.
func (s *HTTPClassifierSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
	if s.paths != nil {
		s.paths.Stop()
	}
}
