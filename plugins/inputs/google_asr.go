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
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("GoogleASRInput", NewGoogleASR)
}

// googleLanguageCodeMap maps friendly language names to the BCP-47 codes accepted
// by the Google ASR service.
var googleLanguageCodeMap = map[string]string{
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

// GoogleASRConfig configures the local microphone-sourced Google ASR sensor.
type GoogleASRConfig struct {
	APIKey               string   `json:"api_key"`
	APIVersion           string   `json:"api_version"`           // "v1" or "v2" (default "v2")
	Rate                 int      `json:"rate"`                  // sample rate Hz (default 48000)
	Chunk                int      `json:"chunk"`                 // frames per buffer (default 4800)
	BaseURL              string   `json:"base_url"`              // override WS endpoint
	MicDeviceIndex       int      `json:"microphone_device_id"`  // -1 = default
	Language             string   `json:"language"`              // default "english"
	AlternativeLanguages []string `json:"alternative_languages"` // v1 only
	EnableTTSInterrupt   bool     `json:"enable_tts_interrupt"`
}

// googleASRParams carries the vendor inputs needed to build a Google asrCommon.
type googleASRParams struct {
	name                 string
	apiKey               string
	apiVersion           string
	baseURL              string
	rate                 int
	language             string
	alternativeLanguages []string
	enableTTSInterrupt   bool
}

// GoogleASRSensor captures audio from a local microphone (via PortAudio) and
// streams it to the Google ASR websocket through the shared asrCommon.
type GoogleASRSensor struct {
	*asrCommon

	cfg        GoogleASRConfig
	paStream   *portaudio.Stream
	audioChunk []int16
}

// NewGoogleASR constructs a GoogleASRSensor with the given configuration.
func NewGoogleASR(configMap map[string]any) (inputs.Sensor, error) {
	cfg := GoogleASRConfig{MicDeviceIndex: -1}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("GoogleASRInput: api_key required")
	}
	if cfg.Rate == 0 {
		cfg.Rate = 48000
	}
	if cfg.Chunk == 0 {
		cfg.Chunk = 4800
	}

	core := newGoogleASRCommon(googleASRParams{
		name:                 "GoogleASRInput",
		apiKey:               cfg.APIKey,
		apiVersion:           cfg.APIVersion,
		baseURL:              cfg.BaseURL,
		rate:                 cfg.Rate,
		language:             cfg.Language,
		alternativeLanguages: cfg.AlternativeLanguages,
		enableTTSInterrupt:   cfg.EnableTTSInterrupt,
	})
	core.log.Info("GoogleASRInput: microphone config", zap.Int("chunk", cfg.Chunk))

	return &GoogleASRSensor{
		asrCommon:  core,
		cfg:        cfg,
		audioChunk: make([]int16, cfg.Chunk),
	}, nil
}

// Listen starts the sensor by connecting to the ASR websocket and running the microphone capture loop.
func (s *GoogleASRSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		if err := providers.PortAudio.Acquire(); err != nil {
			s.log.Error("GoogleASRInput: portaudio init failed", zap.Error(err))
			return
		}

		if err := s.connect(); err != nil {
			s.log.Error("GoogleASRInput: ws connect failed", zap.Error(err))
			return
		}

		if err := s.openMic(ctx); err != nil {
			s.log.Error("GoogleASRInput: mic open failed", zap.Error(err))
			return
		}

		s.pollLoop(ctx, out)
	}()
	return out, nil
}

// Stop signals the capture loop to stop, waits for it to finish, and cleans up resources.
func (s *GoogleASRSensor) Stop() {
	first, captureDone := s.markStopped()
	if !first {
		return
	}

	s.log.Info("GoogleASRInput: stopping sensor")

	s.waitCapture(captureDone)
	s.closeWS()
	providers.PortAudio.Release()
	s.closeZenoh()

	s.log.Info("GoogleASRInput: sensor stopped")
}

// openMic initializes PortAudio, opens the configured microphone stream, and starts the capture loop.
func (s *GoogleASRSensor) openMic(ctx context.Context) error {
	var device *portaudio.DeviceInfo
	var err error

	if s.cfg.MicDeviceIndex >= 0 {
		devices, err := portaudio.Devices()
		if err != nil {
			return fmt.Errorf("GoogleASRInput: list devices: %w", err)
		}

		if s.cfg.MicDeviceIndex >= len(devices) {
			return fmt.Errorf("GoogleASRInput: device index %d out of range", s.cfg.MicDeviceIndex)
		}

		device = devices[s.cfg.MicDeviceIndex]
	} else {
		device, err = portaudio.DefaultInputDevice()

		if err != nil {
			return fmt.Errorf("GoogleASRInput: default device: %w", err)
		}
	}

	s.log.Info("GoogleASRInput: microphone",
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
		return fmt.Errorf("GoogleASRInput: open stream: %w", err)
	}

	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return fmt.Errorf("GoogleASRInput: start stream: %w", err)
	}

	s.mu.Lock()
	s.paStream = stream
	s.captureDone = make(chan struct{})
	s.mu.Unlock()

	go s.captureLoop(ctx, stream)
	go s.statsLoop(ctx)
	s.log.Info("GoogleASRInput: microphone started")
	return nil
}

func (s *GoogleASRSensor) captureLoop(ctx context.Context, stream *portaudio.Stream) {
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
			s.log.Warn("GoogleASRInput: read error", zap.Error(err))
		}

		if providers.Speaking.Load() {
			continue
		}

		pcm := make([]byte, len(s.audioChunk)*2)
		for i, sample := range s.audioChunk {
			binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
		}

		s.sendChunk(pcm)
	}
}

// newGoogleASRCommon resolves Google-specific config and builds the shared asrCommon with the Google parser.
func newGoogleASRCommon(p googleASRParams) *asrCommon {
	return newASRCommon(resolveGoogleASRConfig(p))
}

// resolveGoogleASRConfig maps Google vendor parameters to a transcriberStream config.
func resolveGoogleASRConfig(p googleASRParams) asrCommonConfig {
	apiVersion := strings.TrimSpace(strings.ToLower(p.apiVersion))
	if apiVersion != "v1" && apiVersion != "v2" {
		apiVersion = "v2"
	}

	language := strings.TrimSpace(strings.ToLower(p.language))
	if language == "" {
		language = "english"
	}
	languageCode, ok := googleLanguageCodeMap[language]
	if !ok {
		languageCode = "en-US"
	}

	var altCodes []string
	if apiVersion == "v1" {
		for _, alt := range p.alternativeLanguages {
			alt = strings.TrimSpace(strings.ToLower(alt))
			if code, ok := googleLanguageCodeMap[alt]; ok {
				altCodes = append(altCodes, code)
			}
		}
	}

	wsURL := p.baseURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://api.openmind.com/api/core/google/asr/%s?api_key=%s", apiVersion, p.apiKey)
	}

	return asrCommonConfig{
		Name:               p.name,
		Model:              "google",
		APIVersion:         apiVersion,
		WSURL:              wsURL,
		Rate:               p.rate,
		Language:           language,
		LanguageCode:       languageCode,
		AltCodes:           altCodes,
		EnableTTSInterrupt: p.enableTTSInterrupt,
		ParseMessage:       googleParseMessage,
	}
}

// googleParseMessage implements the Google ASR protocol (speech_start/speech_end/end_of_utterance events plus asr_reply) and records its latency metrics.
func googleParseMessage(s *transcriberStream, msg ASRMessage) string {
	switch msg.Type {
	case "speech_start":
		s.speechStartTime = time.Now()
		s.speechStarted = true
	case "speech_end":
		if s.speechStarted {
			s.observeASR(metrics.ASRSpeechDuration, metrics.ASRSpeechDurationLast, time.Since(s.speechStartTime))
		}
	case "end_of_utterance":
		if s.speechStarted {
			s.observeASR(metrics.ASRUtteranceEndLatency, metrics.ASRUtteranceEndLatencyLast, time.Since(s.speechStartTime))
		}
	}

	if msg.ASRReply == "" || !acceptASRTranscript(msg.ASRReply) {
		return ""
	}

	if s.speechStarted {
		s.observeASR(metrics.ASRLatency, metrics.ASRLatencyLast, time.Since(s.speechStartTime))
		s.speechStarted = false
	}
	return msg.ASRReply
}
