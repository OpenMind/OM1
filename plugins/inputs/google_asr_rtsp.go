package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
)

func init() {
	inputs.Register("GoogleASRRTSPInput", NewGoogleASRRTSP)
}

const rtspReconnectDelay = 2 * time.Second

// GoogleASRRTSPConfig configures the RTSP-sourced Google ASR sensor.
type GoogleASRRTSPConfig struct {
	APIKey               string   `json:"api_key"`
	APIVersion           string   `json:"api_version"`           // "v1" or "v2" (default "v2")
	RTSPURL              string   `json:"rtsp_url"`              // RTSP audio source
	Rate                 int      `json:"rate"`                  // sample rate Hz (default 16000)
	Chunk                int      `json:"chunk"`                 // samples per chunk (default 1600)
	BaseURL              string   `json:"base_url"`              // override WS endpoint
	Language             string   `json:"language"`              // default "english"
	AlternativeLanguages []string `json:"alternative_languages"` // v1 only
	EnableTTSInterrupt   bool     `json:"enable_tts_interrupt"`
}

// GoogleASRRTSPSensor streams audio from an RTSP URL (decoded via ffmpeg) and
// forwards PCM to the Google ASR websocket through the shared googleASRCommon.
type GoogleASRRTSPSensor struct {
	*googleASRCommon

	cfg GoogleASRRTSPConfig
}

// NewGoogleASRRTSP constructs a GoogleASRRTSPSensor with the given configuration.
func NewGoogleASRRTSP(configMap map[string]any) (inputs.Sensor, error) {
	var cfg GoogleASRRTSPConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("GoogleASRRTSPInput: api_key required")
	}
	if cfg.RTSPURL == "" {
		cfg.RTSPURL = "rtsp://localhost:8554/audio"
	}
	if cfg.Rate == 0 {
		cfg.Rate = 16000
	}
	if cfg.Chunk == 0 {
		cfg.Chunk = 1600
	}

	core := NewGoogleASRCommon("GoogleASRRTSPInput", googleASRCommonConfig{
		APIKey:               cfg.APIKey,
		APIVersion:           cfg.APIVersion,
		BaseURL:              cfg.BaseURL,
		Rate:                 cfg.Rate,
		Language:             cfg.Language,
		AlternativeLanguages: cfg.AlternativeLanguages,
	})
	core.log.Info("GoogleASRRTSPInput: rtsp config",
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.Int("chunk", cfg.Chunk),
	)

	return &GoogleASRRTSPSensor{
		googleASRCommon: core,
		cfg:             cfg,
	}, nil
}

// Listen starts the sensor by connecting to the ASR websocket and running the RTSP capture loop.
func (s *GoogleASRRTSPSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		if err := s.wsClient.Connect(); err != nil {
			s.log.Error("GoogleASRRTSPInput: ws connect failed", zap.Error(err))
			return
		}

		s.mu.Lock()
		s.captureDone = make(chan struct{})
		s.mu.Unlock()
		go s.captureLoop(ctx)
		go s.statsLoop(ctx)

		s.pollLoop(ctx, out)
	}()
	return out, nil
}

// Stop signals the capture loop to stop, waits for it to finish, and cleans up resources.
func (s *GoogleASRRTSPSensor) Stop() {
	first, captureDone := s.markStopped()
	if !first {
		return
	}

	s.log.Info("GoogleASRRTSPInput: stopping sensor")

	s.waitCapture(captureDone)
	s.closeWS()
	s.closeZenoh()

	s.log.Info("GoogleASRRTSPInput: sensor stopped")
}

// captureLoop runs the RTSP stream and reconnects on failure until ctx is cancelled.
func (s *GoogleASRRTSPSensor) captureLoop(ctx context.Context) {
	defer func() {
		s.mu.Lock()
		done := s.captureDone
		s.mu.Unlock()
		if done != nil {
			close(done)
		}
	}()

	for {
		if ctx.Err() != nil {
			return
		}

		if err := s.streamRTSP(ctx); err != nil && ctx.Err() == nil {
			s.log.Warn("GoogleASRRTSPInput: RTSP audio error", zap.Error(err))
			s.log.Info("GoogleASRRTSPInput: reconnecting", zap.Duration("delay", rtspReconnectDelay))
			if !util.Sleep(ctx, rtspReconnectDelay) {
				return
			}
		}
	}
}

// streamRTSP runs ffmpeg to capture PCM audio from the RTSP URL and
// sends it to the ASR websocket until an error occurs or ctx is cancelled.
func (s *GoogleASRRTSPSensor) streamRTSP(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-rtsp_transport", "tcp",
		"-i", s.cfg.RTSPURL,
		"-vn",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"-ac", "1",
		"-ar", strconv.Itoa(s.cfg.Rate),
		"-loglevel", "error",
		"pipe:1",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	s.log.Info("GoogleASRRTSPInput: RTSP audio stream connected", zap.String("rtsp_url", s.cfg.RTSPURL))

	chunkBytes := s.cfg.Chunk * 2 // int16 samples
	buf := make([]byte, chunkBytes)
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if _, err := io.ReadFull(stdout, buf); err != nil {
			return fmt.Errorf("read pcm: %w", err)
		}

		if providers.Speaking.Load() && !s.cfg.EnableTTSInterrupt {
			continue
		}

		pcm := make([]byte, chunkBytes)
		copy(pcm, buf)
		s.sendChunk(pcm)
	}
}
