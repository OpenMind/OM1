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
	// PresentAs "disposition": the result is a state the robot carries, not news. Each result sets the dynamic
	// variable "disposition" (which the fuser places in the system part, so every reply is coloured by it), and an
	// event is emitted only when the disposition changes, worded by Events[symbol] (default: "your disposition
	// changed to <label>"). ChangeAfter consecutive results of the same kind are needed before the disposition
	// changes (default 1), so one stray vote does not flip it.
	PresentAs   string            `json:"present_as"`
	ChangeAfter int               `json:"change_after"`
	Events      map[string]string `json:"events"`
	Silent      bool              `json:"silent"` // disposition mode: set the variable, never raise an event
	// Smooth: the disposition is the majority of the last Smooth results (default 1 = the latest result), with
	// "strongly" when they all agree and "faintly" on a bare majority, so that per-vote noise does not reach the
	// prompt and the classifier reads as the slow input it is meant to be.
	Smooth int `json:"smooth"`
	// StrongConfidence: a result at or above this confidence counts as change_after results on its own (default
	// 0.9: with three voters behind the endpoint that is a unanimous vote).
	StrongConfidence float64 `json:"strong_confidence"`
	// While a conversation is in progress (the input ConversationKey, default "Voice", updated within
	// ConversationWindowS seconds, default 30) a disposition change is worded by ConversationEvents[symbol]
	// instead of Events[symbol], so that it reads as something happening to the person the robot is talking
	// with ("the person you are talking with turned away") rather than a new arrival to greet. Default wording:
	// "while you were talking, " + the event text. ConversationWindowS < 0 disables the distinction.
	ConversationWindowS float64           `json:"conversation_window_s"`
	ConversationKey     string            `json:"conversation_key"`
	ConversationEvents  map[string]string `json:"conversation_events"`
	// ReactionCooldownS: no disposition-change event within this many seconds of the runtime acting on any
	// event (default 8; < 0 disables), so one happening seen by two detectors draws one reaction.
	ReactionCooldownS float64 `json:"reaction_cooldown_s"`
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
	lastKind string   // last result kind seen ("positive", "negative", "none"); drives only_on_change
	dispKind string   // disposition mode: the kind currently in force
	dispRun  string   // disposition mode: the kind seen on the last result
	dispN    int      // disposition mode: how many consecutive results of dispRun
	dispWin  []string // disposition mode: the last Smooth kinds, for the majority
	lastText string   // source "input": the last text a label was derived from, so one caption is polled once
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

		// Watch the source often, but send at most once per period: with source "input" a change in the caption is
		// then sent within a quarter second of arriving instead of at the next tick of a 2 s clock.
		watch := s.period
		if s.cfg.Source == "input" && watch > 250*time.Millisecond {
			watch = 250 * time.Millisecond
		}
		ticker := time.NewTicker(watch)
		defer ticker.Stop()
		var lastSent time.Time

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			if time.Since(lastSent) < s.period {
				continue
			}

			raw, err := s.Poll(ctx)
			if raw != nil || err != nil {
				lastSent = time.Now()
			}
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
	if s.cfg.PresentAs == "disposition" {
		return s.disposition(r), nil
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

// disposition handles a result in "disposition" mode: it updates the dynamic variable the fuser shows in the
// system part on every call, and returns an event only when the disposition changes (after change_after
// consecutive results of the new kind). A positive below min_confidence counts as the previous kind.
func (s *HTTPClassifierSensor) disposition(r *httpClassifierResponse) *inputs.Message {
	kind := ""
	switch r.Label {
	case "A":
		if r.Confidence >= s.cfg.MinConfidence {
			kind = "A"
		}
	case "B":
		kind = "B"
	}
	if kind == "" {
		return nil
	}
	strength := ""
	switch {
	case r.Confidence >= 0.9:
		strength = ", strongly"
	case r.Confidence < 0.65:
		strength = ", faintly"
	}
	if n := s.cfg.Smooth; n > 1 {
		// majority of the last n results replaces the single result; strength from how united they are
		s.mu.Lock()
		s.dispWin = append(s.dispWin, kind)
		if len(s.dispWin) > n {
			s.dispWin = s.dispWin[len(s.dispWin)-n:]
		}
		a := 0
		for _, k := range s.dispWin {
			if k == "A" {
				a++
			}
		}
		total := len(s.dispWin)
		s.mu.Unlock()
		b := total - a
		if a >= b {
			kind = "A"
		} else {
			kind = "B"
		}
		margin := a - b
		if margin < 0 {
			margin = -margin
		}
		switch {
		case margin == total:
			strength = ", strongly" // unanimous window
		case margin <= 1:
			strength = ", faintly" // bare majority
		default:
			strength = ""
		}
	}
	label := kind
	if name, ok := s.cfg.Labels[kind]; ok && name != "" {
		label = name
	}
	need := s.cfg.ChangeAfter
	if need < 1 {
		need = 1
	}

	strong := s.cfg.StrongConfidence
	if strong <= 0 {
		strong = 0.9
	}

	s.mu.Lock()
	if kind == s.dispRun {
		s.dispN++
	} else {
		s.dispRun, s.dispN = kind, 1
	}
	if r.Confidence >= strong {
		s.dispN = need
	}
	changed := kind != s.dispKind && s.dispN >= need
	if changed {
		s.dispKind = kind
	}
	current := s.dispKind
	s.mu.Unlock()

	if current == kind {
		providers.IO().SetDynamicVar("disposition", label+strength)
	}
	if !changed || s.cfg.Silent {
		return nil
	}
	if cd := s.cfg.ReactionCooldownS; cd >= 0 {
		if cd == 0 {
			cd = 8
		}
		if providers.IO().SinceReaction() < time.Duration(cd*float64(time.Second)) {
			return nil
		}
	}
	text := "your disposition changed to " + label
	if ev, ok := s.cfg.Events[kind]; ok && ev != "" {
		text = ev
	}
	if s.inConversation() {
		if ev, ok := s.cfg.ConversationEvents[kind]; ok && ev != "" {
			text = ev
		} else {
			text = "while you were talking, " + text
		}
	}
	msg := inputs.NewMessage(text)
	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > httpClassifierMaxMessages {
		s.messages = s.messages[len(s.messages)-httpClassifierMaxMessages:]
	}
	s.mu.Unlock()
	return msg
}

// inConversation reports whether the conversation input was updated within conversation_window_s.
func (s *HTTPClassifierSensor) inConversation() bool {
	quiet := s.cfg.ConversationWindowS
	if quiet == 0 {
		quiet = 30
	}
	if quiet < 0 {
		return false
	}
	key := s.cfg.ConversationKey
	if key == "" {
		key = "Voice"
	}
	in := providers.IO().GetInput(key)
	return in != nil && !in.Timestamp.IsZero() && time.Since(in.Timestamp) < time.Duration(quiet*float64(time.Second))
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
