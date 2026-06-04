//go:build integration

package integration

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/runtime"
)

const (
	mockInputType    = "mock_input"
	scriptedLLMType  = "scripted_llm"
	recordingConnKey = "recording"
)

func init() {
	inputs.Register(mockInputType, newMockSensor)
	llm.Register(scriptedLLMType, newScriptedLLM)
	actions.Register(recordingConnKey, newRecordingConnector)
}

type recordedCall struct {
	Label string
	Args  map[string]any
}

type sink struct {
	mu    sync.Mutex
	calls []recordedCall
}

func (s *sink) add(c recordedCall) {
	s.mu.Lock()
	s.calls = append(s.calls, c)
	s.mu.Unlock()
}

func (s *sink) snapshot() []recordedCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]recordedCall, len(s.calls))
	copy(out, s.calls)
	return out
}

func (s *sink) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

var (
	sinksMu sync.Mutex
	sinks   = map[string]*sink{}

	sessionCounter int64
)

func newSession() (string, *sink) {
	id := "sess-" + itoa(atomic.AddInt64(&sessionCounter, 1))
	s := &sink{}
	sinksMu.Lock()
	sinks[id] = s
	sinksMu.Unlock()
	return id, s
}

func sinkFor(id string) *sink {
	sinksMu.Lock()
	defer sinksMu.Unlock()
	s, ok := sinks[id]
	if !ok {
		s = &sink{}
		sinks[id] = s
	}
	return s
}

type mockSensor struct {
	buffer         string
	messages       []string
	triggers       bool
	publishContext map[string]any
}

func newMockSensor(cfg map[string]any) (inputs.Sensor, error) {
	s := &mockSensor{triggers: true}
	if v, ok := cfg["triggers_tick"].(bool); ok {
		s.triggers = v
	}

	if pc, ok := cfg["publish_context"].(map[string]any); ok {
		s.publishContext = pc
	}

	if text, ok := cfg["text"].(string); ok && text != "" {
		s.messages = append(s.messages, text)
	}

	for _, m := range asStringSlice(cfg["messages"]) {
		s.messages = append(s.messages, m)
	}

	if buf, ok := cfg["buffer"].(string); ok && buf != "" {
		s.buffer = buf
	} else {
		s.buffer = strings.Join(s.messages, " ")
	}

	return s, nil
}

func (s *mockSensor) Listen(ctx context.Context) (<-chan any, error) {
	ch := make(chan any)
	go func() {
		if len(s.publishContext) > 0 {
			providers.ModeContext().Publish(s.publishContext)
		}

		for _, m := range s.messages {
			select {
			case ch <- m:
			case <-ctx.Done():
				return
			}
		}
		<-ctx.Done()
	}()
	return ch, nil
}

func (s *mockSensor) Poll(context.Context) (any, error) { return nil, nil }

func (s *mockSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, _ := raw.(string)
	return inputs.NewMessage(text), nil
}

func (s *mockSensor) FormattedLatestBuffer() string { return s.buffer }
func (s *mockSensor) Stop()                         {}
func (s *mockSensor) TriggersTick() bool            { return s.triggers }

type llmRule struct {
	whenContains string
	toolCalls    []llm.ToolCall
}

type scriptedLLM struct {
	rules   []llmRule
	schemas []map[string]any
}

func newScriptedLLM(cfg map[string]any) (llm.LLM, error) {
	m := &scriptedLLM{}
	for _, raw := range asSlice(cfg["rules"]) {
		rule, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		r := llmRule{}
		r.whenContains, _ = rule["when_contains"].(string)
		for _, rawCall := range asSlice(rule["tool_calls"]) {
			call, ok := rawCall.(map[string]any)
			if !ok {
				continue
			}

			name, _ := call["name"].(string)
			args, _ := call["arguments"].(map[string]any)
			if args == nil {
				args = map[string]any{}
			}

			r.toolCalls = append(r.toolCalls, llm.ToolCall{Name: name, Arguments: args})
		}

		m.rules = append(m.rules, r)
	}

	return m, nil
}

func (m *scriptedLLM) Call(_ context.Context, prompt string, _ []llm.Message) (*llm.Response, error) {
	for _, r := range m.rules {
		if r.whenContains == "" || strings.Contains(prompt, r.whenContains) {
			return &llm.Response{ToolCalls: r.toolCalls}, nil
		}

	}

	return &llm.Response{}, nil
}

func (m *scriptedLLM) SetSchemas(s []map[string]any)     { m.schemas = s }
func (m *scriptedLLM) FunctionSchemas() []map[string]any { return m.schemas }

type recordingConnector struct {
	label string
	sink  *sink
}

func newRecordingConnector(cfg map[string]any) (actions.Connector, error) {
	session, _ := cfg["session"].(string)
	label, _ := cfg["label"].(string)
	return &recordingConnector{label: label, sink: sinkFor(session)}, nil
}

func (c *recordingConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, _ := input.(map[string]any)
	c.sink.add(recordedCall{Label: c.label, Args: args})
	return nil, nil
}

func (c *recordingConnector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(5 * time.Millisecond):
	}
}

func (c *recordingConnector) Stop() {}

func runRuntime(cfg *config.SystemConfig, session string, s *sink, waitForCalls int, timeout time.Duration) []recordedCall {
	injectActionMeta(cfg, session)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rt := runtime.New(cfg, zap.NewNop(), runtime.Options{})

	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		_ = rt.Run(ctx)
	}()

	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	poll := time.NewTicker(10 * time.Millisecond)
	defer poll.Stop()

	for waitForCalls <= 0 || s.len() < waitForCalls {
		select {
		case <-deadline.C:
			goto stop
		case <-poll.C:
		}
	}

stop:
	cancel()
	select {
	case <-runDone:
	case <-time.After(20 * time.Second):
	}
	return s.snapshot()
}

func injectActionMeta(cfg *config.SystemConfig, session string) {
	stamp := func(specs []config.ActionSpec) {
		for i := range specs {
			if specs[i].Config == nil {
				specs[i].Config = map[string]any{}
			}
			label := specs[i].LLMLabel
			if label == "" {
				label = specs[i].Name
			}
			specs[i].Config["session"] = session
			specs[i].Config["label"] = label
		}
	}
	stamp(cfg.AgentActions)
	for name, mode := range cfg.Modes {
		stamp(mode.AgentActions)
		cfg.Modes[name] = mode
	}
}

func asSlice(v any) []any {
	s, _ := v.([]any)
	return s
}

func asStringSlice(v any) []string {
	raw, _ := v.([]any)
	out := make([]string, 0, len(raw))
	for _, e := range raw {
		if s, ok := e.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}

	var buf [20]byte
	i := len(buf)
	neg := n < 0
	if neg {
		n = -n
	}

	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}

	if neg {
		i--
		buf[i] = '-'
	}

	return string(buf[i:])
}
