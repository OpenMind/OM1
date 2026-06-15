package luma_checkin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/luma"
	video "github.com/openmind/om1/internal/providers/vlm"
)

const (
	scannerName        = "LumaCheckin"
	scannerRTSPName    = "LumaCheckinRTSP"
	scannerDescriptor  = "Luma Check-In"
	scannerMaxMessages = 8
	scanChannelBuffer  = 4

	defaultLumaTimeout = 4 * time.Second
)

func init() {
	inputs.Register(scannerName, NewLumaCheckin)
	inputs.Register(scannerRTSPName, NewLumaCheckinRTSP)
}

type LumaConfig struct {
	APIKey                string  `json:"api_key"`
	BaseURL               string  `json:"base_url"`
	EventAPIID            string  `json:"event_api_id"`
	RequestTimeoutSeconds float64 `json:"request_timeout_seconds"`
	SessionKey            string  `json:"session_key"`
}

// Config holds the JSON configuration for the LumaCheckin input plugin.
type Config struct {
	CameraIndex         int         `json:"camera_index"`
	RTSPURL             string      `json:"rtsp_url"`
	CaptureFPS          int         `json:"capture_fps"`
	DecodeFPS           int         `json:"decode_fps"`
	Width               int         `json:"resolution_width"`
	Height              int         `json:"resolution_height"`
	JPEGQuality         int         `json:"jpeg_quality"`
	DedupeWindowSeconds float64     `json:"dedupe_window_seconds"`
	Luma                *LumaConfig `json:"luma"`
}

type frameSource interface {
	Start(ctx context.Context) <-chan video.Frame
	Stop()
}

type guestLookup interface {
	GetGuest(ctx context.Context, pk string) (*luma.Guest, error)
	CheckIn(ctx context.Context, guest *luma.Guest) error
}

type sensor struct {
	name      string
	cfg       Config
	log       *zap.Logger
	source    frameSource
	debouncer *debouncer

	luma            guestLookup
	lumaTimeout     time.Duration
	expectedEventID string

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
	cancel   context.CancelFunc
}

// NewLumaCheckin constructs a camera-backed Luma check-in sensor.
func NewLumaCheckin(configMap map[string]any) (inputs.Sensor, error) {
	cfg := parseConfig(configMap)

	log := logger.Get().Named(scannerName)
	log.Info("initializing",
		zap.Int("camera_index", cfg.CameraIndex),
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.Int("capture_fps", cfg.CaptureFPS),
		zap.Int("decode_fps", cfg.DecodeFPS),
		zap.Int("width", cfg.Width),
		zap.Int("height", cfg.Height),
		zap.Float64("dedupe_window_seconds", cfg.DedupeWindowSeconds),
	)

	source := video.NewVideoStream(video.VideoStreamConfig{
		DeviceIndex: cfg.CameraIndex,
		FPS:         cfg.CaptureFPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
	})
	return newSensor(cfg, log, source), nil
}

// NewLumaCheckinRTSP constructs an RTSP-backed Luma check-in sensor.
func NewLumaCheckinRTSP(configMap map[string]any) (inputs.Sensor, error) {
	cfg := parseConfig(configMap)
	if cfg.RTSPURL == "" {
		cfg.RTSPURL = "rtsp://localhost:8554/top_camera_raw"
	}

	log := logger.Get().Named(scannerRTSPName)
	log.Info("initializing",
		zap.Int("camera_index", cfg.CameraIndex),
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.Int("capture_fps", cfg.CaptureFPS),
		zap.Int("decode_fps", cfg.DecodeFPS),
		zap.Int("width", cfg.Width),
		zap.Int("height", cfg.Height),
		zap.Float64("dedupe_window_seconds", cfg.DedupeWindowSeconds),
	)

	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:     cfg.RTSPURL,
		FPS:         cfg.CaptureFPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
	})
	return newSensor(cfg, log, source), nil
}

func newSensor(cfg Config, log *zap.Logger, source frameSource) *sensor {
	window := time.Duration(cfg.DedupeWindowSeconds * float64(time.Second))
	s := &sensor{
		name:      log.Name(),
		cfg:       cfg,
		log:       log,
		source:    source,
		debouncer: newDebouncer(window),
	}

	if cfg.Luma != nil && cfg.Luma.APIKey != "" && cfg.Luma.EventAPIID != "" {
		timeout := time.Duration(cfg.Luma.RequestTimeoutSeconds * float64(time.Second))
		if timeout <= 0 {
			timeout = defaultLumaTimeout
		}
		var opts []func(*luma.Client)
		if cfg.Luma.SessionKey != "" {
			opts = append(opts, luma.WithSessionKey(cfg.Luma.SessionKey))
		}
		s.luma = luma.NewClient(cfg.Luma.BaseURL, cfg.Luma.APIKey, cfg.Luma.EventAPIID, timeout, opts...)
		s.lumaTimeout = timeout
		s.expectedEventID = cfg.Luma.EventAPIID
		log.Info("luma lookup enabled",
			zap.String("event_api_id", cfg.Luma.EventAPIID),
			zap.Duration("timeout", timeout),
		)
	}

	return s
}

func parseConfig(configMap map[string]any) Config {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.CaptureFPS <= 0 {
		cfg.CaptureFPS = 15
	}
	if cfg.DecodeFPS <= 0 {
		cfg.DecodeFPS = 5
	}
	if cfg.DecodeFPS > cfg.CaptureFPS {
		cfg.DecodeFPS = cfg.CaptureFPS
	}
	if cfg.Width <= 0 {
		cfg.Width = 640
	}
	if cfg.Height <= 0 {
		cfg.Height = 480
	}
	if cfg.JPEGQuality <= 0 {
		cfg.JPEGQuality = 60
	}
	if cfg.DedupeWindowSeconds <= 0 {
		cfg.DedupeWindowSeconds = 30
	}
	return cfg
}

func (s *sensor) Listen(ctx context.Context) (<-chan any, error) {
	ctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()

	frames := s.source.Start(ctx)
	out := make(chan any, scanChannelBuffer)

	stride := s.cfg.CaptureFPS / s.cfg.DecodeFPS
	if stride < 1 {
		stride = 1
	}

	go func() {
		defer close(out)
		defer s.Stop()

		var counter uint64
		for {
			select {
			case <-ctx.Done():
				return
			case frame, ok := <-frames:
				if !ok {
					return
				}
				counter++
				if counter%uint64(stride) != 0 {
					continue
				}

				text, err := decodeQR(frame.JPEG)
				if err != nil {
					if !errors.Is(err, errQRNotFound) {
						s.log.Debug("decode error", zap.Error(err))
					}
					continue
				}
				eventID, pk, valid := parseLumaCheckinURL(text)
				if !valid {
					s.log.Debug("ignoring non-luma qr", zap.String("text", truncate(text, 80)))
					continue
				}
				if !s.debouncer.TryRecord(pk) {
					s.log.Debug("debounced", zap.String("pk", pk))
					continue
				}

				msg := s.formatScanMessage(ctx, pk, eventID)
				s.log.Info("emitted scan", zap.String("pk", pk), zap.String("event", eventID))
				select {
				case out <- msg:
				default:
					s.log.Warn("scan channel full, dropping", zap.String("pk", pk))
				}
			}
		}
	}()

	return out, nil
}

func (s *sensor) Poll(context.Context) (any, error) { return nil, nil }

// RawToText converts a raw scan event into a timestamped Message and appends it
// to the bounded in-memory history.
func (s *sensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}
	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > scannerMaxMessages {
		s.messages = s.messages[len(s.messages)-scannerMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the newest scan formatted for the LLM prompt
// and clears the history. Returns "" when empty.
func (s *sensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", scannerDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(s.name, latest.Message, ts)
	s.messages = nil

	return result
}

// TriggersTick opts the scanner into waking the cortex loop on every fresh scan.
func (s *sensor) TriggersTick() bool { return true }

func (s *sensor) Stop() {
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// formatScanMessage performs the Luma guest lookup and check-in.
func (s *sensor) formatScanMessage(ctx context.Context, pk, eventID string) string {
	if s.luma == nil {
		return fmt.Sprintf("qr_scan: pk=%s event=%s", pk, eventID)
	}

	if s.expectedEventID != "" && eventID != "" && eventID != s.expectedEventID {
		s.log.Info("scan event mismatch",
			zap.String("expected", s.expectedEventID),
			zap.String("got", eventID),
		)
		return fmt.Sprintf("qr_scan_failed: pk=%s reason=event_mismatch", pk)
	}

	lookupCtx, cancel := context.WithTimeout(ctx, s.lumaTimeout)
	defer cancel()

	guest, err := s.luma.GetGuest(lookupCtx, pk)
	if err != nil {
		switch {
		case errors.Is(err, luma.ErrNotFound):
			s.log.Info("luma lookup: not found", zap.String("pk", pk))
			return fmt.Sprintf("qr_scan_failed: pk=%s reason=guest_not_registered", pk)
		case errors.Is(err, luma.ErrUnauthorized):
			s.log.Error("luma lookup: unauthorized")
			return fmt.Sprintf("qr_scan_failed: pk=%s reason=luma_auth", pk)
		default:
			s.log.Warn("luma lookup failed", zap.String("pk", pk), zap.Error(err))
			return fmt.Sprintf("qr_scan_failed: pk=%s reason=lookup_error", pk)
		}
	}
	if guest == nil {
		return fmt.Sprintf("qr_scan_failed: pk=%s reason=empty_response", pk)
	}

	name := luma.FirstName(guest)

	if err := s.luma.CheckIn(ctx, guest); err != nil {
		s.log.Warn("luma check-in failed", zap.String("pk", pk), zap.Error(err))
	} else {
		s.log.Info("luma check-in ok", zap.String("pk", pk), zap.String("name", name))
	}

	luma.RecordCheckIn(name, time.Now())

	s.log.Info("luma lookup ok", zap.String("pk", pk), zap.String("name", name))
	return fmt.Sprintf("qr_scan: name=%s", name)
}
