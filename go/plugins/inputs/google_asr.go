package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gordonklaus/portaudio"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/ws"
)

func init() {
	inputs.Register("GoogleASRInput", NewGoogleASR)
}

var languageCodeMap = map[string]string{
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

var cjkRegex = regexp.MustCompile(`[\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}]`)

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

type ASRMessage struct {
	Type     string `json:"type"`
	ASRReply string `json:"asr_reply"`
}

type AudioMetadata struct {
	Rate                     int      `json:"rate"`
	LanguageCode             string   `json:"language_code"`
	AlternativeLanguageCodes []string `json:"alternative_language_codes,omitempty"`
	Timestamp                int64    `json:"timestamp"`
}

type ASRStatistics struct {
	TotalChunksSent uint64
	TotalBytesSent  uint64
	FailedChunks    uint64
	LastSendTime    time.Time
	mu              sync.RWMutex
}

type GoogleASRSensor struct {
	cfg          GoogleASRConfig
	log          *zap.Logger
	languageCode string
	altCodes     []string
	apiVersion   string

	wsClient   *ws.Client
	paStream   *portaudio.Stream
	audioChunk []int16

	// transcriptCh is used to send accepted transcripts from the WS callback to the main loop.
	transcriptCh chan string

	// messages accumulates transcripts between fuser ticks
	messages []string

	mu              sync.Mutex
	stopped         bool
	speechStartTime time.Time
	speechStarted   bool

	stats ASRStatistics
}

func NewGoogleASR(configMap map[string]any) (inputs.Sensor, error) {
	var cfg GoogleASRConfig
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
	if cfg.MicDeviceIndex == 0 {
		cfg.MicDeviceIndex = -1
	}

	apiVersion := strings.TrimSpace(strings.ToLower(cfg.APIVersion))
	if apiVersion != "v1" && apiVersion != "v2" {
		apiVersion = "v2"
	}
	language := strings.TrimSpace(strings.ToLower(cfg.Language))
	if language == "" {
		language = "english"
	}
	languageCode, ok := languageCodeMap[language]
	if !ok {
		languageCode = "en-US"
	}
	var altCodes []string
	if apiVersion == "v1" {
		for _, alt := range cfg.AlternativeLanguages {
			alt = strings.TrimSpace(strings.ToLower(alt))
			if code, ok := languageCodeMap[alt]; ok {
				altCodes = append(altCodes, code)
			}
		}
	}

	wsURL := cfg.BaseURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://api.openmind.com/api/core/google/asr/%s?api_key=%s",
			apiVersion, cfg.APIKey)
	}

	log, _ := zap.NewProduction()
	log.Info("GoogleASRInput: initializing",
		zap.String("language", language),
		zap.String("language_code", languageCode),
		zap.String("api_version", apiVersion),
		zap.Int("rate", cfg.Rate),
		zap.Int("chunk", cfg.Chunk),
	)

	s := &GoogleASRSensor{
		cfg:          cfg,
		log:          log,
		languageCode: languageCode,
		altCodes:     altCodes,
		apiVersion:   apiVersion,
		audioChunk:   make([]int16, cfg.Chunk),
		transcriptCh: make(chan string, 32),
	}
	s.wsClient = ws.New(ws.Config{URL: wsURL, Reconnect: true}, log, s.onWSMessage)
	return s, nil
}

func (s *GoogleASRSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)

		if err := portaudio.Initialize(); err != nil {
			s.log.Error("GoogleASRInput: portaudio init failed", zap.Error(err))
			return
		}
		if err := s.wsClient.Connect(); err != nil {
			portaudio.Terminate()
			s.log.Error("GoogleASRInput: ws connect failed", zap.Error(err))
			return
		}
		if err := s.openMic(ctx); err != nil {
			s.wsClient.Close()
			portaudio.Terminate()
			s.log.Error("GoogleASRInput: mic open failed", zap.Error(err))
			return
		}

		for {
			raw, err := s.Poll(ctx)
			if err != nil {
				return
			}
			select {
			case out <- raw:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

func (s *GoogleASRSensor) Poll(ctx context.Context) (any, error) {
	select {
	case text, ok := <-s.transcriptCh:
		if !ok {
			return nil, context.Canceled
		}
		return text, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *GoogleASRSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)

	if !ok || text == "" {
		return nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		s.messages = append(s.messages, text)
	} else {
		s.messages[len(s.messages)-1] = s.messages[len(s.messages)-1] + " " + text
	}

	return inputs.NewMessage(text), nil
}

func (s *GoogleASRSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\nVoice: %q\n", latest)

	s.log.Info("GoogleASRInput: flushing buffer", zap.String("text", latest))
	s.messages = nil

	return result
}

func (s *GoogleASRSensor) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return
	}

	s.stopped = true
	s.log.Info("GoogleASRInput: stopping sensor")

	if s.paStream != nil {
		_ = s.paStream.Stop()
		_ = s.paStream.Close()
	}

	if s.wsClient != nil {
		s.wsClient.Close()
	}

	portaudio.Terminate()

	s.log.Info("GoogleASRInput: sensor stopped")
}

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
	s.mu.Unlock()

	go s.captureLoop(ctx)
	go s.statsLoop(ctx)
	s.log.Info("GoogleASRInput: microphone started")
	return nil
}

func (s *GoogleASRSensor) captureLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := s.paStream.Read(); err != nil && err.Error() != "Input overflowed" {
			s.log.Warn("GoogleASRInput: read error", zap.Error(err))
		}

		pcm := make([]byte, len(s.audioChunk)*2)
		for i, sample := range s.audioChunk {
			binary.LittleEndian.PutUint16(pcm[i*2:], uint16(sample))
		}

		packet, err := s.packageAudio(pcm)
		if err != nil {
			s.log.Warn("GoogleASRInput: package error", zap.Error(err))
			continue
		}
		if err := s.wsClient.Send(packet); err != nil {
			s.log.Warn("GoogleASRInput: send error", zap.Error(err))
			s.stats.mu.Lock()
			s.stats.FailedChunks++
			s.stats.mu.Unlock()
		} else {
			s.stats.mu.Lock()
			s.stats.TotalChunksSent++
			s.stats.TotalBytesSent += uint64(len(packet))
			s.stats.LastSendTime = time.Now()
			s.stats.mu.Unlock()
		}
	}
}

func (s *GoogleASRSensor) statsLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			s.PrintStatistics()
			return
		case <-ticker.C:
			s.PrintStatistics()
		}
	}
}

func (s *GoogleASRSensor) packageAudio(pcm []byte) ([]byte, error) {
	meta := AudioMetadata{
		Rate:                     s.cfg.Rate,
		LanguageCode:             s.languageCode,
		AlternativeLanguageCodes: s.altCodes,
		Timestamp:                time.Now().UnixMilli(),
	}
	headerBytes, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	hLen := len(headerBytes)
	packet := make([]byte, 4+hLen+len(pcm))
	binary.BigEndian.PutUint32(packet[0:4], uint32(hLen))
	copy(packet[4:4+hLen], headerBytes)
	copy(packet[4+hLen:], pcm)
	return packet, nil
}

func (s *GoogleASRSensor) onWSMessage(msgType int, data []byte) {
	if msgType != websocket.TextMessage {
		return
	}
	var msg ASRMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return
	}

	switch msg.Type {
	case "speech_start":
		s.speechStartTime = time.Now()
		s.speechStarted = true
		s.log.Info("GoogleASRInput: speech start", zap.Time("time", s.speechStartTime))
	case "speech_end":
		if s.speechStarted {
			s.log.Info("GoogleASRInput: speech end",
				zap.Duration("duration", time.Since(s.speechStartTime)))
		}
	case "end_of_utterance":
		if s.speechStarted {
			s.log.Info("GoogleASRInput: end of utterance",
				zap.Duration("latency", time.Since(s.speechStartTime)))
		}
	}

	if msg.ASRReply == "" || !s.acceptTranscript(msg.ASRReply) {
		return
	}

	var latency time.Duration
	if s.speechStarted {
		latency = time.Since(s.speechStartTime)
		s.speechStarted = false
	}
	s.log.Info("GoogleASRInput: transcript accepted",
		zap.String("text", msg.ASRReply),
		zap.Duration("asr_latency", latency),
	)

	select {
	case s.transcriptCh <- msg.ASRReply:
	default:
		s.log.Warn("GoogleASRInput: transcript buffer full, dropping",
			zap.String("text", msg.ASRReply))
	}
}

func (s *GoogleASRSensor) acceptTranscript(text string) bool {
	if cjkRegex.MatchString(text) {
		return utf8.RuneCountInString(text) > 2
	}

	return len(strings.Fields(text)) > 1
}

func (s *GoogleASRSensor) PrintStatistics() {
	s.stats.mu.RLock()
	totalChunks := s.stats.TotalChunksSent
	totalBytes := s.stats.TotalBytesSent
	failed := s.stats.FailedChunks
	lastSendTime := s.stats.LastSendTime
	s.stats.mu.RUnlock()

	now := time.Now()
	fields := []zap.Field{
		zap.Uint64("total_chunks_sent", totalChunks),
		zap.String("total_bytes_mb", fmt.Sprintf("%.2f", float64(totalBytes)/(1024*1024))),
		zap.Uint64("failed_chunks", failed),
	}

	if !lastSendTime.IsZero() {
		fields = append(fields, zap.Duration("time_since_last_send", now.Sub(lastSendTime)))
	}

	s.log.Info("GoogleASRInput: statistics", fields...)
}
