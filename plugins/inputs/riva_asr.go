package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/gordonklaus/portaudio"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("RivaASRInput", NewRivaASR)
}

// rivaLanguageCodeMap maps friendly language names to the BCP-47 codes accepted
// by the Riva ASR service.
var rivaLanguageCodeMap = map[string]string{
	"english":    "en-US",
	"chinese":    "cmn-Hans-CN",
	"german":     "de-DE",
	"french":     "fr-FR",
	"japanese":   "ja-JP",
	"korean":     "ko-KR",
	"spanish":    "es-ES",
	"italian":    "it-IT",
	"portuguese": "pt-BR",
	"russian":    "ru-RU",
	"arabic":     "ar-SA",
}

// rivaAPIVersion is the fixed api_version label used for Riva ASR metrics.
const rivaAPIVersion = "v1"

// rivaDefaultBaseURL is the default Riva ASR websocket endpoint.
const rivaDefaultBaseURL = "ws://localhost:6790"

// RivaASRConfig configures the local microphone-sourced Riva ASR sensor.
type RivaASRConfig struct {
	APIKey             string `json:"api_key"`
	Rate               int    `json:"rate"`                 // sample rate Hz (default 48000)
	Chunk              int    `json:"chunk"`                // frames per buffer (default 12144)
	BaseURL            string `json:"base_url"`             // override WS endpoint
	MicDeviceIndex     int    `json:"microphone_device_id"` // -1 = default
	Language           string `json:"language"`             // default "english"
	EnableTTSInterrupt bool   `json:"enable_tts_interrupt"`
}

// rivaASRParams carries the vendor inputs needed to build a Riva asrCommon.
type rivaASRParams struct {
	name               string
	baseURL            string
	rate               int
	language           string
	enableTTSInterrupt bool
}

// RivaASRSensor captures audio from a local microphone (via PortAudio) and
// streams it to the Riva ASR websocket through the shared asrCommon.
type RivaASRSensor struct {
	*asrCommon

	cfg        RivaASRConfig
	paStream   *portaudio.Stream
	audioChunk []int16
}

// NewRivaASR constructs a RivaASRSensor with the given configuration.
func NewRivaASR(configMap map[string]any) (inputs.Sensor, error) {
	cfg := RivaASRConfig{MicDeviceIndex: -1}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.Rate == 0 {
		cfg.Rate = 48000
	}
	if cfg.Chunk == 0 {
		cfg.Chunk = 12144
	}

	core := newRivaASRCommon(rivaASRParams{
		name:               "RivaASRInput",
		baseURL:            cfg.BaseURL,
		rate:               cfg.Rate,
		language:           cfg.Language,
		enableTTSInterrupt: cfg.EnableTTSInterrupt,
	})
	core.log.Info("RivaASRInput: microphone config", zap.Int("chunk", cfg.Chunk))

	return &RivaASRSensor{
		asrCommon:  core,
		cfg:        cfg,
		audioChunk: make([]int16, cfg.Chunk),
	}, nil
}

// Listen starts the sensor by connecting to the ASR websocket and running the microphone capture loop.
func (s *RivaASRSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		if err := providers.PortAudio.Acquire(); err != nil {
			s.log.Error("RivaASRInput: portaudio init failed", zap.Error(err))
			return
		}

		if err := s.connect(); err != nil {
			s.log.Error("RivaASRInput: ws connect failed", zap.Error(err))
			return
		}

		if err := s.openMic(ctx); err != nil {
			s.log.Error("RivaASRInput: mic open failed", zap.Error(err))
			return
		}

		s.pollLoop(ctx, out)
	}()
	return out, nil
}

// Stop signals the capture loop to stop, waits for it to finish, and cleans up resources.
func (s *RivaASRSensor) Stop() {
	first, captureDone := s.markStopped()
	if !first {
		return
	}

	s.log.Info("RivaASRInput: stopping sensor")

	s.waitCapture(captureDone)
	s.closeWS()
	providers.PortAudio.Release()
	s.closeZenoh()

	s.log.Info("RivaASRInput: sensor stopped")
}

// openMic initializes PortAudio, opens the configured microphone stream, and starts the capture loop.
func (s *RivaASRSensor) openMic(ctx context.Context) error {
	var device *portaudio.DeviceInfo
	var err error

	if s.cfg.MicDeviceIndex >= 0 {
		devices, err := portaudio.Devices()
		if err != nil {
			return fmt.Errorf("RivaASRInput: list devices: %w", err)
		}

		if s.cfg.MicDeviceIndex >= len(devices) {
			return fmt.Errorf("RivaASRInput: device index %d out of range", s.cfg.MicDeviceIndex)
		}

		device = devices[s.cfg.MicDeviceIndex]
	} else {
		device, err = portaudio.DefaultInputDevice()

		if err != nil {
			return fmt.Errorf("RivaASRInput: default device: %w", err)
		}
	}

	s.log.Info("RivaASRInput: microphone",
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
		return fmt.Errorf("RivaASRInput: open stream: %w", err)
	}

	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return fmt.Errorf("RivaASRInput: start stream: %w", err)
	}

	s.mu.Lock()
	s.paStream = stream
	s.captureDone = make(chan struct{})
	s.mu.Unlock()

	go s.captureLoop(ctx, stream)
	go s.statsLoop(ctx)
	s.log.Info("RivaASRInput: microphone started")
	return nil
}

func (s *RivaASRSensor) captureLoop(ctx context.Context, stream *portaudio.Stream) {
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
			s.log.Warn("RivaASRInput: read error", zap.Error(err))
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

// newRivaASRCommon resolves Riva-specific config and builds the shared asrCommon with the Riva parser.
func newRivaASRCommon(p rivaASRParams) *asrCommon {
	return newASRCommon(resolveRivaASRConfig(p))
}

// resolveRivaASRConfig maps Riva vendor parameters to a transcriberStream config.
func resolveRivaASRConfig(p rivaASRParams) asrCommonConfig {
	language := strings.TrimSpace(strings.ToLower(p.language))
	if language == "" {
		language = "english"
	}
	languageCode, ok := rivaLanguageCodeMap[language]
	if !ok {
		languageCode = "en-US"
	}

	wsURL := p.baseURL
	if wsURL == "" {
		wsURL = rivaDefaultBaseURL
	}

	return asrCommonConfig{
		Name:               p.name,
		Model:              "riva",
		APIVersion:         rivaAPIVersion,
		WSURL:              wsURL,
		Rate:               p.rate,
		Language:           language,
		LanguageCode:       languageCode,
		EnableTTSInterrupt: p.enableTTSInterrupt,
		ParseMessage:       rivaParseMessage,
	}
}

// acceptRivaTranscript reports whether a Riva transcript is long enough to keep.
func acceptRivaTranscript(text string) bool {
	return len(strings.Fields(text)) > 2
}

// rivaParseMessage simply forwards any reply that clears the length threshold.
func rivaParseMessage(_ *transcriberStream, msg ASRMessage) string {
	if msg.ASRReply == "" || !acceptRivaTranscript(msg.ASRReply) {
		return ""
	}
	return msg.ASRReply
}
