package inputs

// HTTPClassifierInput polls a local signal, sends it to an external HTTP classification service, and
// surfaces the service's response as an input.
//
// Config:
//
//	{ type: "HTTPClassifierInput", config: { endpoint_url: "http://127.0.0.1:5557", period_ms: 2000,
//	                                         source: "paths", min_confidence: 0.5, only_on_change: true } }
//
// only_on_change reports a result only when it differs from the last reported kind (positive / negative / none),
// so a condition that persists across polls produces one message per episode rather than one per poll.
//
// source "input" takes the label from another input's latest text (input_key, e.g. "VLMGemini") through
// symbol_rules, e.g. [{match: ["waving", "smiling"], symbol: "A"}, {match: ["turned away"], symbol: "B"}];
// a text matching no rule, or one already classified, makes no request. labels names the results in
// messages, e.g. {A: "warmth", B: "distance"}.

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
	Source         string  `json:"source"`     // "paths" | "file" | "input"
	LabelFile      string  `json:"label_file"` // source "file"
	ReportNegative bool    `json:"report_negative"`
	MinConfidence  float64 `json:"min_confidence"`
	OnlyOnChange   bool    `json:"only_on_change"` // report a result only when it differs from the last reported one
	// source "input": derive the label from the latest text another input stored on the IO provider under
	// InputKey (an input's name, e.g. a VLM caption), using SymbolRules. No rule matching means no poll.
	InputKey    string       `json:"input_key"`
	SymbolRules []SymbolRule `json:"symbol_rules"`
	// Labels names the results in messages, e.g. {"A": "warmth", "B": "distance"}; unset keeps the generic text.
	Labels map[string]string `json:"labels"`
}

// SymbolRule maps phrases found in an input's text to a symbol. Matching is case-insensitive substring; the
// first rule whose phrase occurs wins, in config order.
type SymbolRule struct {
	Match  []string `json:"match"`
	Symbol string   `json:"symbol"`
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
	lastKind string // last result kind seen ("positive", "negative", "none"); drives only_on_change
	lastText string // source "input": the last text a label was derived from, so one caption is polled once
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
	if cfg.Source != "paths" && cfg.Source != "file" && cfg.Source != "input" {
		return nil, fmt.Errorf("http_classifier: source must be \"paths\", \"file\" or \"input\", got %q", cfg.Source)
	}
	if cfg.Source == "file" && cfg.LabelFile == "" {
		return nil, fmt.Errorf("http_classifier: source \"file\" needs label_file")
	}
	if cfg.Source == "input" && (cfg.InputKey == "" || len(cfg.SymbolRules) == 0) {
		return nil, fmt.Errorf("http_classifier: source \"input\" needs input_key and symbol_rules")
	}
	for i, r := range cfg.SymbolRules {
		if r.Symbol != "A" && r.Symbol != "B" {
			return nil, fmt.Errorf("http_classifier: symbol_rules[%d]: symbol must be A or B, got %q", i, r.Symbol)
		}
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

// label derives the current A/B label from the configured source. An empty label with a nil error means
// "nothing to classify this period" (source "input" with no matching rule, or an unchanged text).
func (s *HTTPClassifierSensor) label() (string, error) {
	switch s.cfg.Source {
	case "input":
		in := providers.IO().GetInput(s.cfg.InputKey)
		if in == nil || strings.TrimSpace(in.Input) == "" {
			return "", nil
		}
		s.mu.Lock()
		same := in.Input == s.lastText
		s.lastText = in.Input
		s.mu.Unlock()
		if same {
			return "", nil
		}
		return matchSymbol(in.Input, s.cfg.SymbolRules), nil
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

// matchSymbol returns the symbol of the first rule with a phrase occurring in text (case-insensitive), or "".
func matchSymbol(text string, rules []SymbolRule) string {
	t := strings.ToLower(text)
	for _, r := range rules {
		for _, m := range r.Match {
			if m != "" && strings.Contains(t, strings.ToLower(m)) {
				return r.Symbol
			}
		}
	}
	return ""
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
	if label == "" {
		return nil, nil // nothing to classify this period; no request is made
	}
	return s.classify(ctx, label)
}

// RawToText turns a response into a message.
func (s *HTTPClassifierSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	r, ok := raw.(*httpClassifierResponse)
	if !ok || r == nil {
		return nil, nil
	}
	// Classify the result; a positive below min_confidence counts as "none" so a weak A does not end an episode.
	kind, text := "none", "external classifier returned no result"
	switch r.Label {
	case "A":
		if r.Confidence >= s.cfg.MinConfidence {
			kind, text = "positive", s.describe("A", "confirms change", r)
		}
	case "B":
		kind, text = "negative", s.describe("B", "reports no change", r)
	}

	s.mu.Lock()
	changed := kind != s.lastKind
	s.lastKind = kind
	s.mu.Unlock()

	if r.Label == "A" && kind == "none" {
		return nil, nil
	}
	if kind != "positive" && !s.cfg.ReportNegative {
		return nil, nil
	}
	if s.cfg.OnlyOnChange && !changed {
		return nil, nil
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

// describe renders a result for the prompt: the configured label name if any, else the generic phrase.
func (s *HTTPClassifierSensor) describe(symbol, generic string, r *httpClassifierResponse) string {
	if name, ok := s.cfg.Labels[symbol]; ok && name != "" {
		return fmt.Sprintf("external classifier reports: %s (%.2f, %.0f ms)", name, r.Confidence, r.LatencyMs)
	}
	return fmt.Sprintf("external classifier %s (%.2f, %.0f ms)", generic, r.Confidence, r.LatencyMs)
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

// Role: a classifier result is an event.
func (s *HTTPClassifierSensor) Role() inputs.Role { return inputs.RoleEvent }

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
