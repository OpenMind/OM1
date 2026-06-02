package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"

	"github.com/gordonklaus/portaudio"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("GoogleASRInput", NewGoogleASR)
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

// GoogleASRSensor captures audio from a local microphone (via PortAudio) and
// streams it to the Google ASR websocket through the shared googleASRCommon.
type GoogleASRSensor struct {
	*googleASRCommon

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

	core := NewGoogleASRCommon("GoogleASRInput", googleASRCommonConfig{
		APIKey:               cfg.APIKey,
		APIVersion:           cfg.APIVersion,
		BaseURL:              cfg.BaseURL,
		Rate:                 cfg.Rate,
		Language:             cfg.Language,
		AlternativeLanguages: cfg.AlternativeLanguages,
	})
	core.log.Info("GoogleASRInput: microphone config", zap.Int("chunk", cfg.Chunk))

	return &GoogleASRSensor{
		googleASRCommon: core,
		cfg:             cfg,
		audioChunk:      make([]int16, cfg.Chunk),
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

		if err := s.wsClient.Connect(); err != nil {
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
