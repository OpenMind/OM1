package vlm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	video "github.com/openmind/om1/internal/providers/vlm"
)

func init() {
	inputs.Register("VLMOpenAI", NewVLMOpenAI)
	inputs.Register("VLMOpenAIRTSP", NewVLMOpenAIRTSP)
}

const (
	vlmDescriptor  = "Vision"
	vlmMaxMessages = 10
	defaultFPS     = 10
	defaultRTSPURL = "rtsp://localhost:8556/raw"
)

type providerDefaults struct {
	baseURL   string
	apiKey    string
	model     string
	prompt    string
	maxTokens int
	extraBody map[string]any
	timeout   time.Duration
	warmup    bool
}

var openAIDefaults = providerDefaults{
	baseURL:   "https://api.openmind.com/api/core/openai",
	model:     "gpt-4o-mini",
	prompt:    "What is the most interesting aspect in this series of images?",
	maxTokens: 300,
}

// frameSource abstracts the source of video frames, whether from a camera or an RTSP stream, allowing vlmSensor to process them uniformly.
type frameSource interface {
	Start(ctx context.Context) <-chan video.Frame
	Stop()
}

type VLMConfig struct {
	APIKey      string `json:"api_key"`
	BaseURL     string `json:"base_url"`
	Model       string `json:"model"`
	Prompt      string `json:"prompt"`
	FPS         int    `json:"fps"`
	MaxTokens   int    `json:"max_tokens"`
	JPEGQuality int    `json:"jpeg_quality"`
	Width       int    `json:"resolution_width"`
	Height      int    `json:"resolution_height"`

	ExtraBody  map[string]any `json:"extra_body"`
	TimeoutSec float64        `json:"timeout_sec"`
	Warmup     bool           `json:"warmup"`

	// EventMarker: when a description contains this marker (e.g. "[change:"), the text after it (to the closing
	// "]" or the end) is raised as an event, so the cortex reacts to a change in view; descriptions without it
	// stay world state. The prompt may contain {previous}, replaced by the previous description, so the model can
	// judge what changed. EventMinIntervalS (default 8) is the least time between two such events.
	EventMarker       string  `json:"event_marker"`
	EventMinIntervalS float64 `json:"event_min_interval_s"`

	// Camera-only.
	CameraIndex int `json:"camera_index"`

	// RTSP-only.
	RTSPURL string `json:"rtsp_url"`
}

// vlmSensor implements the inputs.Sensor interface, capturing video frames, sending them to a vision API for description, and emitting the descriptions as messages.
type vlmSensor struct {
	name      string
	log       *zap.Logger
	describer *video.Describer
	source    frameSource

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
	cancel   context.CancelFunc

	prompt      string // the configured prompt, with {previous} unexpanded
	previous    string // the last description, for {previous}
	eventMarker string
	eventMinGap time.Duration
	pendingEv   string
	lastEventAt time.Time
}

// NewVLMOpenAI constructs a camera-backed OpenAI VLM sensor.
func NewVLMOpenAI(configMap map[string]any) (inputs.Sensor, error) {
	return NewCameraSensor("VLMOpenAI", openAIDefaults, configMap)
}

// NewVLMOpenAIRTSP constructs an RTSP-backed OpenAI VLM sensor.
func NewVLMOpenAIRTSP(configMap map[string]any) (inputs.Sensor, error) {
	return NewRTSPSensor("VLMOpenAIRTSP", openAIDefaults, configMap)
}

// NewCameraSensor builds a camera-backed VLM sensor for the given backend.
func NewCameraSensor(name string, defaults providerDefaults, configMap map[string]any) (inputs.Sensor, error) {
	cfg, err := parseConfig(configMap, defaults)
	if err != nil {
		return nil, err
	}
	source := video.NewVideoStream(video.VideoStreamConfig{
		DeviceIndex: cfg.CameraIndex,
		FPS:         cfg.FPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
	})
	return NewSensor(name, cfg, source), nil
}

// NewRTSPSensor builds an RTSP-backed VLM sensor for the given backend.
func NewRTSPSensor(name string, defaults providerDefaults, configMap map[string]any) (inputs.Sensor, error) {
	cfg, err := parseConfig(configMap, defaults)
	if err != nil {
		return nil, err
	}
	if cfg.RTSPURL == "" {
		cfg.RTSPURL = defaultRTSPURL
	}
	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:     cfg.RTSPURL,
		FPS:         cfg.FPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
	})
	return NewSensor(name, cfg, source), nil
}

func parseConfig(configMap map[string]any, defaults providerDefaults) (VLMConfig, error) {
	var cfg VLMConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		cfg.APIKey = defaults.apiKey
	}
	if cfg.APIKey == "" {
		return cfg, fmt.Errorf("vlm: api_key is required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaults.baseURL
	}
	if cfg.Model == "" {
		cfg.Model = defaults.model
	}
	if cfg.Prompt == "" {
		cfg.Prompt = defaults.prompt
	}
	if cfg.FPS <= 0 {
		cfg.FPS = defaultFPS
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = defaults.maxTokens
	}
	if cfg.ExtraBody == nil {
		cfg.ExtraBody = defaults.extraBody
	}
	if cfg.TimeoutSec <= 0 {
		cfg.TimeoutSec = defaults.timeout.Seconds()
	}
	cfg.Warmup = cfg.Warmup || defaults.warmup
	return cfg, nil
}

func NewSensor(name string, cfg VLMConfig, source frameSource) *vlmSensor {
	log := logger.Get().Named(name)
	log.Info("initializing",
		zap.String("base_url", cfg.BaseURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
	)
	gap := time.Duration(cfg.EventMinIntervalS * float64(time.Second))
	if cfg.EventMinIntervalS <= 0 {
		gap = 8 * time.Second
	}
	return &vlmSensor{
		name:        name,
		log:         log,
		source:      source,
		prompt:      cfg.Prompt,
		eventMarker: cfg.EventMarker,
		eventMinGap: gap,
		describer: video.NewDescriber(video.Describer{
			Name:      name,
			APIKey:    cfg.APIKey,
			BaseURL:   cfg.BaseURL,
			Model:     cfg.Model,
			Prompt:    cfg.Prompt,
			MaxTokens: cfg.MaxTokens,
			ExtraBody: cfg.ExtraBody,
			Timeout:   time.Duration(cfg.TimeoutSec * float64(time.Second)),
			Warmup:    cfg.Warmup,
			Log:       log,
		}),
	}
}

// Listen starts the frame capture and description loop, emitting each description as it arrives.
func (s *vlmSensor) Listen(ctx context.Context) (<-chan any, error) {
	ctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()

	frames := s.source.Start(ctx)
	out := make(chan any)

	go func() {
		defer close(out)
		defer s.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case frame, ok := <-frames:
				if !ok {
					return
				}

				providers.LatestFrame().Set(frame.JPEG, frame.Timestamp)
				if strings.Contains(s.prompt, "{previous}") {
					s.mu.Lock()
					prev := s.previous
					s.mu.Unlock()
					if prev == "" {
						prev = "(none yet)"
					}
					s.describer.Prompt = strings.ReplaceAll(s.prompt, "{previous}", prev)
				}
				text, err := s.describer.Describe(ctx, base64.StdEncoding.EncodeToString(frame.JPEG))
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					s.log.Warn("vision request failed", zap.Error(err))
					continue
				}
				if text == "" {
					continue
				}

				select {
				case out <- text:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return out, nil
}

func (s *vlmSensor) Poll(context.Context) (any, error) {
	return nil, nil
}

// RawToText appends a new description to the message buffer, returning it as an inputs.Message for downstream processing.
func (s *vlmSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.previous = text
	s.messages = append(s.messages, *msg)
	if len(s.messages) > vlmMaxMessages {
		s.messages = s.messages[len(s.messages)-vlmMaxMessages:]
	}
	if ev := s.eventIn(text); ev != "" && time.Since(s.lastEventAt) >= s.eventMinGap && providers.IO().SinceReaction() >= s.eventMinGap {
		s.pendingEv = ev
		s.lastEventAt = time.Now()
	}
	s.mu.Unlock()

	return msg, nil
}

// eventIn returns the event text marked in a description, or "".
func (s *vlmSensor) eventIn(text string) string {
	if s.eventMarker == "" {
		return ""
	}
	i := strings.Index(text, s.eventMarker)
	if i < 0 {
		return ""
	}
	ev := text[i+len(s.eventMarker):]
	if j := strings.IndexAny(ev, "[]"); j >= 0 { // stop at the next tag, whichever way round the model put them
		ev = ev[:j]
	}
	return strings.TrimSpace(ev)
}

// LatestEvent returns the pending change event once (inputs.EventSource). While a conversation is in progress
// (the "Voice" input updated within 30 s) the event is worded as being about the person the robot is talking
// with, so that a change in view is never read as a new arrival to greet.
func (s *vlmSensor) LatestEvent() string {
	s.mu.Lock()
	ev := s.pendingEv
	s.pendingEv = ""
	s.mu.Unlock()
	if ev == "" {
		return ""
	}
	if in := providers.IO().GetInput("Voice"); in != nil && !in.Timestamp.IsZero() && time.Since(in.Timestamp) < 30*time.Second {
		return fmt.Sprintf("%s (the person you are talking with): %s", vlmDescriptor, ev)
	}
	return fmt.Sprintf("%s: %s", vlmDescriptor, ev)
}

// TriggersTick: a sensor that can raise events wakes the cortex when a description arrives (an idle tick costs
// nothing with skip_idle_ticks); a pure world-state sensor does not.
func (s *vlmSensor) TriggersTick() bool { return s.eventMarker != "" }

// FormattedLatestBuffer returns the most recent description in the buffer, prefixed by the descriptor and suffixed by a newline.
func (s *vlmSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", vlmDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(s.name, latest.Message, ts)
	s.messages = nil

	return result
}

// Role: the state of the surroundings, kept for awareness; never by itself a reason to speak.
func (s *vlmSensor) Role() inputs.Role { return inputs.RoleWorld }

func (s *vlmSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}

	s.stopped = true
	cancel := s.cancel
	s.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	s.source.Stop()
	s.log.Info("stopping sensor")
}
