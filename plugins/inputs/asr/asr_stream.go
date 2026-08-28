package asr

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/ws"
)

// AudioMetadata is the JSON header prepended to each audio chunk sent to the ASR websocket.
type AudioMetadata struct {
	Rate                     int      `json:"rate"`
	LanguageCode             string   `json:"language_code"`
	AlternativeLanguageCodes []string `json:"alternative_language_codes,omitempty"`
	Timestamp                int64    `json:"timestamp"`
}

// ASRStatistics tracks audio chunk send counters for periodic logging.
type ASRStatistics struct {
	TotalChunksSent uint64
	TotalBytesSent  uint64
	FailedChunks    uint64
	LastSendTime    time.Time
	mu              sync.RWMutex
}

// transcriberStream is one vendor ASR connection: it packages and streams PCM to
// a websocket, parses the vendor's protocol, and hands accepted transcripts to onTranscript.
type transcriberStream struct {
	name string
	log  *zap.Logger

	provider     string
	apiVersion   string
	language     string
	languageCode string
	altCodes     []string
	rate         int

	parseMessage asrMessageParser
	wsClient     *ws.Client

	onTranscript func(provider, text string)

	mu              sync.Mutex
	speechStartTime time.Time
	speechEndTime   time.Time
	speechStarted   bool

	stats ASRStatistics
}

// newTranscriberStream builds a stream from a vendor-resolved config, wiring its
// websocket client to onWSMessage and its accepted transcripts to onTranscript.
func newTranscriberStream(cfg asrCommonConfig, log *zap.Logger, onTranscript func(provider, text string)) *transcriberStream {
	s := &transcriberStream{
		name:         cfg.Name,
		log:          log,
		provider:     cfg.Provider,
		apiVersion:   cfg.APIVersion,
		language:     cfg.Language,
		languageCode: cfg.LanguageCode,
		altCodes:     cfg.AltCodes,
		rate:         cfg.Rate,
		parseMessage: cfg.ParseMessage,
		onTranscript: onTranscript,
	}
	s.wsClient = ws.New(ws.Config{URL: cfg.WSURL, Reconnect: true}, log, s.onWSMessage)
	return s
}

// connect dials the ASR websocket.
func (s *transcriberStream) connect() error { return s.wsClient.Connect() }

// closeWS closes the ASR websocket client.
func (s *transcriberStream) closeWS() {
	if s.wsClient != nil {
		s.wsClient.Close()
	}
}

// packageAudio prepends the JSON audio header (length-prefixed) to a PCM chunk.
func (s *transcriberStream) packageAudio(pcm []byte) ([]byte, error) {
	meta := AudioMetadata{
		Rate:                     s.rate,
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

// sendChunk packages and sends a PCM chunk over the websocket, updating statistics.
func (s *transcriberStream) sendChunk(pcm []byte) {
	packet, err := s.packageAudio(pcm)
	if err != nil {
		s.log.Warn("package error", zap.Error(err))
		return
	}

	if err := s.wsClient.Send(packet); err != nil {
		s.log.Warn("send error", zap.Error(err))
		s.stats.mu.Lock()
		s.stats.FailedChunks++
		s.stats.mu.Unlock()
		return
	}

	s.stats.mu.Lock()
	s.stats.TotalChunksSent++
	s.stats.TotalBytesSent += uint64(len(packet))
	s.stats.LastSendTime = time.Now()
	s.stats.mu.Unlock()
}

// speechWindow reports when this utterance was actually spoken. Callers must
// hold s.mu.
//
// The end is not simply now. A transcript lands after the vendor has decided
// the utterance is over and finished decoding it, so "now" trails the last
// word by the endpointing delay -- hundreds of milliseconds, sometimes more.
// Handing that padded window to active-speaker detection charges a person for
// silence they did not make: the scorer sees a stretch with no speech in it,
// and the answer's lifetime, which is derived from how long they spoke, is
// inflated by the same amount. So prefer a real speech-end mark when the
// vendor sends one (Google does; ElevenLabs commits the transcript at the end
// of speech, so now is already tight).
//
// A stale mark is worse than none: if it did not land inside this utterance,
// fall back rather than report a window that ends before it starts.
func (s *transcriberStream) speechWindow() (start, end time.Time) {
	start = s.speechStartTime
	if !s.speechEndTime.IsZero() && s.speechEndTime.After(start) {
		return start, s.speechEndTime
	}
	return start, time.Now()
}

// onWSMessage decodes an ASR websocket message, delegates vendor-specific parsing to parseMessage,
// and forwards any accepted transcript to onTranscript.
func (s *transcriberStream) onWSMessage(msgType int, data []byte) {
	if msgType != websocket.TextMessage {
		return
	}
	var msg ASRMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}

	s.mu.Lock()
	transcript := s.parseMessage(s, msg)
	speechStart, speechEnd := s.speechWindow()
	s.mu.Unlock()

	if transcript == "" {
		return
	}

	providers.Speaker().ResolveAsync(speechStart, speechEnd)

	if s.onTranscript != nil {
		s.onTranscript(s.provider, transcript)
	}
}

// observeASR records an ASR latency metric pair with this stream's labels.
func (s *transcriberStream) observeASR(hist *prometheus.HistogramVec, gauge *prometheus.GaugeVec, d time.Duration) {
	seconds := d.Seconds()
	hist.WithLabelValues(s.provider, s.language, s.apiVersion).Observe(seconds)
	gauge.WithLabelValues(s.provider, s.language, s.apiVersion).Set(seconds)
}

// statsLoop logs send statistics every 15 seconds until ctx is cancelled.
func (s *transcriberStream) statsLoop(ctx context.Context) {
	ticker := time.NewTicker(15 * time.Second)
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

// PrintStatistics logs the cumulative audio send counters.
func (s *transcriberStream) PrintStatistics() {
	s.stats.mu.RLock()
	totalChunks := s.stats.TotalChunksSent
	totalBytes := s.stats.TotalBytesSent
	failed := s.stats.FailedChunks
	lastSendTime := s.stats.LastSendTime
	s.stats.mu.RUnlock()

	fields := []zap.Field{
		zap.Uint64("total_chunks_sent", totalChunks),
		zap.String("total_bytes_mb", fmt.Sprintf("%.2f", float64(totalBytes)/(1024*1024))),
		zap.Uint64("failed_chunks", failed),
	}
	if !lastSendTime.IsZero() {
		fields = append(fields, zap.Duration("time_since_last_send", time.Since(lastSendTime)))
	}

	s.log.Info("statistics", fields...)
}
