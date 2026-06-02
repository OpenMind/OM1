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
	inputs.Register("ElevenLabsASRInput", NewElevenLabsASR)
}

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

// ElevenLabsASRSensor captures audio from a local microphone (via PortAudio) and
// streams it to the ElevenLabs ASR websocket through the shared elevenlabsASRCommon.
type ElevenLabsASRSensor struct {
	*elevenlabsASRCommon

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

	core := NewElevenLabsASRCommon("ElevenLabsASRInput", elevenlabsASRCommonConfig{
		APIKey:   cfg.APIKey,
		BaseURL:  cfg.BaseURL,
		Rate:     cfg.Rate,
		Language: cfg.Language,
	})
	core.log.Info("ElevenLabsASRInput: microphone config", zap.Int("chunk", cfg.Chunk))

	return &ElevenLabsASRSensor{
		elevenlabsASRCommon: core,
		cfg:                 cfg,
		audioChunk:          make([]int16, cfg.Chunk),
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

		if err := s.wsClient.Connect(); err != nil {
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
