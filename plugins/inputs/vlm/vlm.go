package vlm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
)

type providerDefaults struct {
	baseURL   string
	model     string
	prompt    string
	maxTokens int
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

	// Camera-only.
	CameraIndex int `json:"camera_index"`

	// RTSP-only.
	RTSPURL      string `json:"rtsp_url"`
	DecodeFormat string `json:"decode_format"`
}

// vlmSensor implements the inputs.Sensor interface, capturing video frames, sending them to a vision API for description, and emitting the descriptions as messages.
type vlmSensor struct {
	name   string
	log    *zap.Logger
	client *visionClient
	source frameSource

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
	cancel   context.CancelFunc
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
	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:      cfg.RTSPURL,
		DecodeFormat: cfg.DecodeFormat,
		FPS:          cfg.FPS,
		Width:        cfg.Width,
		Height:       cfg.Height,
		JPEGQuality:  cfg.JPEGQuality,
	})
	return NewSensor(name, cfg, source), nil
}

func parseConfig(configMap map[string]any, defaults providerDefaults) (VLMConfig, error) {
	var cfg VLMConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
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
	return cfg, nil
}

func NewSensor(name string, cfg VLMConfig, source frameSource) *vlmSensor {
	log := logger.Get()
	log.Info(name+": initializing",
		zap.String("base_url", cfg.BaseURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
	)
	return &vlmSensor{
		name:   name,
		log:    log,
		source: source,
		client: &visionClient{
			name:      name,
			apiKey:    cfg.APIKey,
			baseURL:   cfg.BaseURL,
			model:     cfg.Model,
			prompt:    cfg.Prompt,
			maxTokens: cfg.MaxTokens,
			log:       log,
		},
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

				text, err := s.client.describe(ctx, base64.StdEncoding.EncodeToString(frame.JPEG))
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					s.log.Warn(s.name+": vision request failed", zap.Error(err))
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
	s.messages = append(s.messages, *msg)
	if len(s.messages) > vlmMaxMessages {
		s.messages = s.messages[len(s.messages)-vlmMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the most recent description in the buffer, prefixed by the descriptor and suffixed by a newline.
func (s *vlmSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%q: %q\n", vlmDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(s.name, latest.Message, ts)
	s.messages = nil

	return result
}

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
	s.log.Info(s.name + ": stopping sensor")
}
