package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gordonklaus/portaudio"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("ElevenLabsASRInput", NewElevenLabsASR)
}

// elevenlabsLanguageCodeMap maps friendly language names to the short codes
// accepted by the ElevenLabs ASR service. "auto" enables language detection.
var elevenlabsLanguageCodeMap = map[string]string{
	"auto":       "auto",
	"english":    "en",
	"spanish":    "es",
	"french":     "fr",
	"german":     "de",
	"italian":    "it",
	"portuguese": "pt",
	"japanese":   "ja",
	"korean":     "ko",
	"chinese":    "zh",
	"dutch":      "nl",
	"polish":     "pl",
	"russian":    "ru",
}

// elevenlabsAPIVersion is the fixed api_version label used for ElevenLabs ASR metrics.
const elevenlabsAPIVersion = "v1"

// ElevenLabsASRConfig configures the local microphone-sourced ElevenLabs ASR sensor.
type ElevenLabsASRConfig struct {
	APIKey             string `json:"api_key"`
	Rate               int    `json:"rate"`                 // sample rate Hz (default 48000)
	Chunk              int    `json:"chunk"`                // frames per buffer (default 4800)
	BaseURL            string `json:"base_url"`             // override WS endpoint
	MicDeviceIndex     int    `json:"microphone_device_id"` // -1 = default
	Language           string `json:"language"`             // default "auto"
	EnableTTSInterrupt bool   `json:"enable_tts_interrupt"`
}

// elevenlabsASRParams carries the vendor inputs needed to build an ElevenLabs asrCommon.
type elevenlabsASRParams struct {
	name               string
	apiKey             string
	baseURL            string
	rate               int
	language           string
	enableTTSInterrupt bool
}

// ElevenLabsASRSensor captures audio from a local microphone (via PortAudio) and
// streams it to the ElevenLabs ASR websocket through the shared asrCommon.
type ElevenLabsASRSensor struct {
	*asrCommon

	cfg        ElevenLabsASRConfig
	paStream   *portaudio.Stream
	audioChunk []int16
}

// NewElevenLabsASR constructs an ElevenLabsASRSensor with the given configuration.
func NewElevenLabsASR(configMap map[string]any) (inputs.Sensor, error) {
	cfg := ElevenLabsASRConfig{MicDeviceIndex: -1}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("ElevenLabsASRInput: api_key required")
	}
	if cfg.Rate == 0 {
		cfg.Rate = 48000
	}
	if cfg.Chunk == 0 {
		cfg.Chunk = 4800
	}

	core := newElevenLabsASRCommon(elevenlabsASRParams{
		name:               "ElevenLabsASRInput",
		apiKey:             cfg.APIKey,
		baseURL:            cfg.BaseURL,
		rate:               cfg.Rate,
		language:           cfg.Language,
		enableTTSInterrupt: cfg.EnableTTSInterrupt,
	})
	core.log.Info("ElevenLabsASRInput: microphone config", zap.Int("chunk", cfg.Chunk))

	return &ElevenLabsASRSensor{
		asrCommon:  core,
		cfg:        cfg,
		audioChunk: make([]int16, cfg.Chunk),
	}, nil
}

// Listen starts the sensor by connecting to the ASR websocket and running the microphone capture loop.
func (s *ElevenLabsASRSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		if err := providers.PortAudio.Acquire(); err != nil {
			s.log.Error("ElevenLabsASRInput: portaudio init failed", zap.Error(err))
			return
		}

		if err := s.connect(); err != nil {
			s.log.Error("ElevenLabsASRInput: ws connect failed", zap.Error(err))
			return
		}

		if err := s.openMic(ctx); err != nil {
			s.log.Error("ElevenLabsASRInput: mic open failed", zap.Error(err))
			return
		}

		s.pollLoop(ctx, out)
	}()
	return out, nil
}

// Stop signals the capture loop to stop, waits for it to finish, and cleans up resources.
func (s *ElevenLabsASRSensor) Stop() {
	first, captureDone := s.markStopped()
	if !first {
		return
	}

	s.log.Info("ElevenLabsASRInput: stopping sensor")

	s.waitCapture(captureDone)
	s.closeWS()
	providers.PortAudio.Release()
	s.closeZenoh()

	s.log.Info("ElevenLabsASRInput: sensor stopped")
}

// openMic initializes PortAudio, opens the configured microphone stream, and starts the capture loop.
func (s *ElevenLabsASRSensor) openMic(ctx context.Context) error {
	var device *portaudio.DeviceInfo
	var err error

	if s.cfg.MicDeviceIndex >= 0 {
		devices, err := portaudio.Devices()
		if err != nil {
			return fmt.Errorf("ElevenLabsASRInput: list devices: %w", err)
		}

		if s.cfg.MicDeviceIndex >= len(devices) {
			return fmt.Errorf("ElevenLabsASRInput: device index %d out of range", s.cfg.MicDeviceIndex)
		}

		device = devices[s.cfg.MicDeviceIndex]
	} else {
		device, err = portaudio.DefaultInputDevice()

		if err != nil {
			return fmt.Errorf("ElevenLabsASRInput: default device: %w", err)
		}
	}

	s.log.Info("ElevenLabsASRInput: microphone",
		zap.String("device", device.Name),
		zap.Int("rate", s.cfg.Rate),
		zap.Int("chunk", s.cfg.Chunk),
		zap.Float64("chunk_ms", float64(s.cfg.Chunk)/float64(s.cfg.Rate)*1000),
	)

	params := portaudio.StreamParameters{
		Input: portaudio.StreamDeviceParameters{
			Device:   device,
			Channels: 1,
			Latency:  device.DefaultHighInputLatency,
		},
		SampleRate:      float64(s.cfg.Rate),
		FramesPerBuffer: s.cfg.Chunk,
	}

	stream, err := portaudio.OpenStream(params, s.audioChunk)
	if err != nil {
		return fmt.Errorf("ElevenLabsASRInput: open stream: %w", err)
	}

	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return fmt.Errorf("ElevenLabsASRInput: start stream: %w", err)
	}

	s.mu.Lock()
	s.paStream = stream
	s.captureDone = make(chan struct{})
	s.mu.Unlock()

	go s.captureLoop(ctx, stream)
	go s.statsLoop(ctx)
	s.log.Info("ElevenLabsASRInput: microphone started")
	return nil
}

func (s *ElevenLabsASRSensor) captureLoop(ctx context.Context, stream *portaudio.Stream) {
	defer func() {
		_ = stream.Stop()
		_ = stream.Close()

		s.mu.Lock()
		if s.paStream == stream {
			s.paStream = nil
		}
		done := s.captureDone
		s.mu.Unlock()

		if done != nil {
			close(done)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := stream.Read(); err != nil && err.Error() != "Input overflowed" {
			s.log.Warn("ElevenLabsASRInput: read error", zap.Error(err))
		}

		if providers.Speaking.Load() && !s.cfg.EnableTTSInterrupt {
			continue
		}

		pcm := make([]byte, len(s.audioChunk)*2)
		for i, sample := range s.audioChunk {
			binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
		}

		s.sendChunk(pcm)
	}
}

// newElevenLabsASRCommon resolves ElevenLabs-specific config and builds the shared asrCommon with the ElevenLabs parser.
func newElevenLabsASRCommon(p elevenlabsASRParams) *asrCommon {
	return newASRCommon(resolveElevenLabsASRConfig(p))
}

// resolveElevenLabsASRConfig maps ElevenLabs vendor parameters to a transcriberStream config.
func resolveElevenLabsASRConfig(p elevenlabsASRParams) asrCommonConfig {
	language := strings.TrimSpace(strings.ToLower(p.language))
	if language == "" {
		language = "auto"
	}

	languageCode, ok := elevenlabsLanguageCodeMap[language]
	if !ok {
		logger.Get().Error(p.name+": unsupported language, defaulting to auto",
			zap.String("language", language))
		language = "auto"
		languageCode = "auto"
	}

	wsURL := p.baseURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://api.openmind.com/api/core/elevenlabs/asr?api_key=%s", p.apiKey)
	}

	return asrCommonConfig{
		Name:               p.name,
		Model:              "elevenlabs",
		APIVersion:         elevenlabsAPIVersion,
		WSURL:              wsURL,
		Rate:               p.rate,
		Language:           language,
		LanguageCode:       languageCode,
		EnableTTSInterrupt: p.enableTTSInterrupt,
		ParseMessage:       elevenlabsParseMessage,
	}
}

// elevenlabsParseMessage implements the ElevenLabs ASR protocol (partial marks speech start, committed carries the transcript) and records its latency metric.
func elevenlabsParseMessage(s *transcriberStream, msg ASRMessage) string {
	if msg.Type == "partial" {
		// Record the start time only on the first partial of an utterance, else latency shrinks to time-since-last-partial.
		if !s.speechStarted {
			s.speechStartTime = time.Now()
			s.speechStarted = true
		}
		return ""
	}

	if msg.Type != "committed" {
		return ""
	}

	// A committed message ends the segment, so reset the timer even when the transcript is dropped.
	speechStarted := s.speechStarted
	speechStartTime := s.speechStartTime
	s.speechStarted = false

	if msg.ASRReply == "" || !acceptASRTranscript(msg.ASRReply) {
		return ""
	}

	if speechStarted {
		s.observeASR(metrics.ASRLatency, metrics.ASRLatencyLast, time.Since(speechStartTime))
	}
	return msg.ASRReply
}
