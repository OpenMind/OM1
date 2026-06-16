package vlm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	video "github.com/openmind/om1/internal/providers/vlm"
	"github.com/openmind/om1/internal/util"
)

func init() {
	bg.Register("VLMOpenAI", NewVLMOpenAI)
	bg.Register("VLMOpenAIRTSP", NewVLMOpenAIRTSP)
}

const (
	defaultFPS     = 10
	defaultRTSPURL = "rtsp://localhost:8554/top_camera_raw"

	vlmRestartDelay = 2 * time.Second
)

// providerDefaults holds the per-backend vision endpoint defaults.
type providerDefaults struct {
	baseURL   string
	model     string
	prompt    string
	maxTokens int
}

var openAIDefaults = providerDefaults{
	baseURL:   "https://api.openmind.com/api/core/openai",
	model:     "gpt-4o-mini",
	prompt:    "In one concise sentence, describe what you see in this image.",
	maxTokens: 300,
}

// frameSource abstracts a camera or RTSP video stream so the background can
// process either uniformly.
type frameSource interface {
	Start(ctx context.Context) <-chan video.Frame
	Stop()
}

// VLMConfig configures a VLM background.
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

	CameraIndex int    `json:"camera_index"`
	RTSPURL     string `json:"rtsp_url"`
}

// NewVLMOpenAI constructs a camera-backed OpenAI VLM background.
func NewVLMOpenAI(configMap map[string]any) (bg.Background, error) {
	return NewCameraBackground("VLMOpenAI", openAIDefaults, configMap)
}

// NewVLMOpenAIRTSP constructs an RTSP-backed OpenAI VLM background.
func NewVLMOpenAIRTSP(configMap map[string]any) (bg.Background, error) {
	return NewRTSPBackground("VLMOpenAIRTSP", openAIDefaults, configMap)
}

// NewCameraBackground builds a camera-backed VLM background for the given backend.
func NewCameraBackground(name string, defaults providerDefaults, configMap map[string]any) (bg.Background, error) {
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
	return NewBackground(name, cfg, source), nil
}

// NewRTSPBackground builds an RTSP-backed VLM background for the given backend.
func NewRTSPBackground(name string, defaults providerDefaults, configMap map[string]any) (bg.Background, error) {
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
	return NewBackground(name, cfg, source), nil
}

// parseConfig decodes the raw config map and applies backend defaults.
func parseConfig(configMap map[string]any, defaults providerDefaults) (VLMConfig, error) {
	var cfg VLMConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return cfg, fmt.Errorf("vlm background: api_key is required")
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

// vlmBackground implements a vision-language model background that captures frames from a source.
type vlmBackground struct {
	name      string
	log       *zap.Logger
	describer *video.Describer
	source    frameSource

	mu      sync.Mutex
	stopped bool
}

// newBackground wires up the frame source and describer for the given config.
func NewBackground(name string, cfg VLMConfig, source frameSource) bg.Background {
	log := logger.Get().Named(name)
	log.Info("initializing background",
		zap.String("base_url", cfg.BaseURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
	)

	return &vlmBackground{
		name:   name,
		log:    log,
		source: source,
		describer: video.NewDescriber(video.Describer{
			Name:      name,
			APIKey:    cfg.APIKey,
			BaseURL:   cfg.BaseURL,
			Model:     cfg.Model,
			Prompt:    cfg.Prompt,
			MaxTokens: cfg.MaxTokens,
			Log:       log,
		}),
	}
}

// Run starts the frame capture and description loop until the context is cancelled.
func (b *vlmBackground) Run(ctx context.Context) {
	frames := b.source.Start(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case frame, ok := <-frames:
			if !ok {
				b.source.Stop()
				util.Sleep(ctx, vlmRestartDelay)
				return
			}

			providers.LatestFrame().Set(frame.JPEG, frame.Timestamp)

			text, err := b.describer.Describe(ctx, base64.StdEncoding.EncodeToString(frame.JPEG))
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				b.log.Warn("vision request failed", zap.Error(err))
				continue
			}
			if text == "" {
				continue
			}

			video.LatestDescription().Set(text, frame.Timestamp)
		}
	}
}

// Stop signals the frame source to stop and prevents further restarts.
func (b *vlmBackground) Stop() {
	b.mu.Lock()
	if b.stopped {
		b.mu.Unlock()
		return
	}
	b.stopped = true
	b.mu.Unlock()

	b.source.Stop()
	b.log.Info("stopping background")
}
