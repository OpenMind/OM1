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

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/ws"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// languageCodeMap maps friendly language names to BCP-47 codes accepted by the
// Google ASR service.
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

// cjkRegex matches CJK characters (Chinese, Japanese kana, Hangul) used to decide
// the minimum-length threshold for accepting a transcript.
var cjkRegex = regexp.MustCompile(`[\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}]`)

const asrZenohTopic = "om/asr/text"

// ASRMessage is a transcript/status message received from the ASR websocket.
type ASRMessage struct {
	Type     string `json:"type"`
	ASRReply string `json:"asr_reply"`
	Time     int64  `json:"time"`
}

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

// googleASRCommonConfig holds the configuration parameters for googleASRCommon.
type googleASRCommonConfig struct {
	APIKey               string
	APIVersion           string
	BaseURL              string
	Rate                 int
	Language             string
	AlternativeLanguages []string
}

// googleASRCommon encapsulates the shared logic for Google ASR sensors.
type googleASRCommon struct {
	name         string // log prefix, e.g. "GoogleASRInput"
	log          *zap.Logger
	rate         int
	language     string // friendly name, used as a metric label
	languageCode string
	altCodes     []string
	apiVersion   string

	wsClient *ws.Client

	// captureDone is closed when the capture loop has fully stopped.
	captureDone chan struct{}

	// transcriptCh carries accepted transcripts from the WS callback to the main loop.
	transcriptCh chan string

	// messages accumulates transcripts between fuser ticks.
	messages []string

	mu              sync.Mutex
	stopped         bool
	speechStartTime time.Time
	speechStarted   bool

	stats ASRStatistics

	zenohSession   *zenohsession.Session
	zenohPublisher *zenohsession.Publisher
}

// NewGoogleASRCommon constructs a googleASRCommon with the given configuration.
func NewGoogleASRCommon(name string, cfg googleASRCommonConfig) *googleASRCommon {
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

	log := logger.Get()
	log.Info(name+": initializing",
		zap.String("language", language),
		zap.String("language_code", languageCode),
		zap.String("api_version", apiVersion),
		zap.Int("rate", cfg.Rate),
	)

	c := &googleASRCommon{
		name:         name,
		log:          log,
		rate:         cfg.Rate,
		language:     language,
		languageCode: languageCode,
		altCodes:     altCodes,
		apiVersion:   apiVersion,
		transcriptCh: make(chan string, 32),
	}

	c.wsClient = ws.New(ws.Config{URL: wsURL, Reconnect: true}, log, c.onWSMessage)

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn(name+": zenoh unavailable, ASR broadcast disabled", zap.Error(err))
	} else {
		pub, err := sess.DeclarePublisher(asrZenohTopic)
		if err != nil {
			sess.Close()
			log.Warn(name+": failed to declare zenoh publisher, ASR broadcast disabled", zap.Error(err))
		} else {
			c.zenohSession = sess
			c.zenohPublisher = pub
			log.Info(name+": zenoh publisher initialized", zap.String("topic", asrZenohTopic))
		}
	}

	return c
}

// Poll returns the next accepted transcript, blocking until one is available or
// ctx is cancelled.
func (c *googleASRCommon) Poll(ctx context.Context) (any, error) {
	select {
	case text, ok := <-c.transcriptCh:
		if !ok {
			return nil, context.Canceled
		}
		return text, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// pollLoop forwards transcripts from Poll onto out until ctx is cancelled or the
// transcript channel closes.
func (c *googleASRCommon) pollLoop(ctx context.Context, out chan any) {
	for {
		raw, err := c.Poll(ctx)
		if err != nil {
			return
		}
		select {
		case out <- raw:
		case <-ctx.Done():
			return
		}
	}
}

// RawToText appends an accepted transcript to the current utterance buffer.
func (c *googleASRCommon) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.messages) == 0 {
		c.messages = append(c.messages, text)
	} else {
		c.messages[len(c.messages)-1] = c.messages[len(c.messages)-1] + " " + text
	}

	return inputs.NewMessage(text), nil
}

// TriggersTick reports that ASR input wakes the cortex loop.
func (c *googleASRCommon) TriggersTick() bool { return true }

// FormattedLatestBuffer returns the buffered utterance as a Voice block, records
// it on the IO provider, broadcasts it over zenoh, and clears the buffer.
func (c *googleASRCommon) FormattedLatestBuffer() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.messages) == 0 {
		return ""
	}

	latest := c.messages[len(c.messages)-1]
	result := fmt.Sprintf("\nVoice: %q\n", latest)

	c.log.Info(c.name+": flushing buffer", zap.String("text", latest))
	c.messages = nil

	providers.IO().AddInput("Voice", latest, time.Time{})

	if c.zenohPublisher != nil {
		payload := serializeASRText(latest)
		if err := c.zenohPublisher.Put(payload); err != nil {
			c.log.Warn(c.name+": zenoh publish failed", zap.Error(err))
		} else {
			c.log.Info(c.name+": published ASR to zenoh", zap.String("text", latest))
		}
	}

	return result
}

// packageAudio prepends the JSON audio header (length-prefixed) to a PCM chunk.
func (c *googleASRCommon) packageAudio(pcm []byte) ([]byte, error) {
	meta := AudioMetadata{
		Rate:                     c.rate,
		LanguageCode:             c.languageCode,
		AlternativeLanguageCodes: c.altCodes,
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
func (c *googleASRCommon) sendChunk(pcm []byte) {
	packet, err := c.packageAudio(pcm)
	if err != nil {
		c.log.Warn(c.name+": package error", zap.Error(err))
		return
	}

	if err := c.wsClient.Send(packet); err != nil {
		c.log.Warn(c.name+": send error", zap.Error(err))
		c.stats.mu.Lock()
		c.stats.FailedChunks++
		c.stats.mu.Unlock()
		return
	}

	c.stats.mu.Lock()
	c.stats.TotalChunksSent++
	c.stats.TotalBytesSent += uint64(len(packet))
	c.stats.LastSendTime = time.Now()
	c.stats.mu.Unlock()
}

// onWSMessage parses ASR websocket messages, records timing metrics, and forwards
// accepted transcripts to the main loop.
func (c *googleASRCommon) onWSMessage(msgType int, data []byte) {
	if msgType != websocket.TextMessage {
		return
	}
	var msg ASRMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.stopped {
		return
	}

	switch msg.Type {
	case "speech_start":
		c.speechStartTime = time.Now()
		c.speechStarted = true
		c.log.Debug(c.name+": speech start",
			zap.Time("receive time", c.speechStartTime),
			zap.Int64("client latency ms", time.Since(time.UnixMilli(msg.Time)).Milliseconds()),
		)
	case "speech_end":
		if c.speechStarted {
			duration := time.Since(c.speechStartTime)
			c.log.Debug(c.name+": speech end",
				zap.Duration("latency", duration),
				zap.Int64("client latency ms", time.Since(time.UnixMilli(msg.Time)).Milliseconds()),
			)
			c.observeASR(metrics.ASRSpeechDuration, metrics.ASRSpeechDurationLast, duration)
		}
	case "end_of_utterance":
		if c.speechStarted {
			latency := time.Since(c.speechStartTime)
			c.log.Debug(c.name+": end of utterance",
				zap.Duration("latency", latency),
				zap.Int64("client latency ms", time.Since(time.UnixMilli(msg.Time)).Milliseconds()),
			)
			c.observeASR(metrics.ASRUtteranceEndLatency, metrics.ASRUtteranceEndLatencyLast, latency)
		}
	}

	if msg.ASRReply == "" || !c.acceptTranscript(msg.ASRReply) {
		return
	}

	var latency time.Duration
	if c.speechStarted {
		latency = time.Since(c.speechStartTime)
		c.speechStarted = false
		c.observeASR(metrics.ASRLatency, metrics.ASRLatencyLast, latency)
	}
	c.log.Info(c.name+": transcript accepted",
		zap.String("text", msg.ASRReply),
		zap.Duration("asr_latency", latency),
	)

	select {
	case c.transcriptCh <- msg.ASRReply:
	default:
		c.log.Warn(c.name+": transcript buffer full, dropping",
			zap.String("text", msg.ASRReply))
	}
}

// observeASR records ASR latency metrics with the appropriate labels for this sensor's configuration.
func (c *googleASRCommon) observeASR(hist *prometheus.HistogramVec, gauge *prometheus.GaugeVec, d time.Duration) {
	seconds := d.Seconds()
	hist.WithLabelValues("google", c.language, c.apiVersion).Observe(seconds)
	gauge.WithLabelValues("google", c.language, c.apiVersion).Set(seconds)
}

// acceptTranscript reports whether a transcript is long enough to keep: more than
// one word, or more than two characters for CJK text.
func (c *googleASRCommon) acceptTranscript(text string) bool {
	if cjkRegex.MatchString(text) {
		return utf8.RuneCountInString(text) > 2
	}
	return len(strings.Fields(text)) > 1
}

// statsLoop logs send statistics every 15 seconds until ctx is cancelled.
func (c *googleASRCommon) statsLoop(ctx context.Context) {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			c.PrintStatistics()
			return
		case <-ticker.C:
			c.PrintStatistics()
		}
	}
}

// PrintStatistics logs the cumulative audio send counters.
func (c *googleASRCommon) PrintStatistics() {
	c.stats.mu.RLock()
	totalChunks := c.stats.TotalChunksSent
	totalBytes := c.stats.TotalBytesSent
	failed := c.stats.FailedChunks
	lastSendTime := c.stats.LastSendTime
	c.stats.mu.RUnlock()

	fields := []zap.Field{
		zap.Uint64("total_chunks_sent", totalChunks),
		zap.String("total_bytes_mb", fmt.Sprintf("%.2f", float64(totalBytes)/(1024*1024))),
		zap.Uint64("failed_chunks", failed),
	}
	if !lastSendTime.IsZero() {
		fields = append(fields, zap.Duration("time_since_last_send", time.Since(lastSendTime)))
	}

	c.log.Info(c.name+": statistics", fields...)
}

// markStopped flips the stopped flag exactly once. It returns whether this call
// was the first to stop the sensor and a snapshot of captureDone to wait on.
func (c *googleASRCommon) markStopped() (firstStop bool, captureDone chan struct{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stopped {
		return false, nil
	}
	c.stopped = true
	return true, c.captureDone
}

// waitCapture waits up to 5s for the capture loop to finish.
func (c *googleASRCommon) waitCapture(captureDone chan struct{}) {
	if captureDone == nil {
		return
	}
	select {
	case <-captureDone:
	case <-time.After(5 * time.Second):
		c.log.Warn(c.name + ": capture loop did not stop within timeout")
	}
}

// closeWS closes the ASR websocket client.
func (c *googleASRCommon) closeWS() {
	if c.wsClient != nil {
		c.wsClient.Close()
	}
}

// closeZenoh drops the zenoh publisher and closes the session.
func (c *googleASRCommon) closeZenoh() {
	if c.zenohPublisher != nil {
		c.zenohPublisher.Drop()
		c.zenohPublisher = nil
		c.log.Info(c.name + ": zenoh publisher dropped")
	}
	if c.zenohSession != nil {
		c.zenohSession.Close()
		c.zenohSession = nil
		c.log.Info(c.name + ": zenoh session closed")
	}
}

// serializeASRText encodes an ASRText message in CDR little-endian format,
// matching the pycdr2 serialization used by the Python google_asr plugins.
//
// Wire layout (offsets from start of buffer):
//
//	[0]   CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]   stamp.sec:    int32 LE   (data offset 0)
//	[8]   stamp.nanosec: uint32 LE (data offset 4)
//	[12]  frame_id:     CDR string (uint32 length + bytes + null, padded to 4-byte data boundary)
//	[...]  text:         CDR string (uint32 length + bytes + null, padded to 4-byte data boundary)
func serializeASRText(text string) []byte {
	now := time.Now()
	frameID := uuid.New().String()

	var buf []byte

	// CDR encapsulation header (little-endian)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)

	// stamp.sec (int32 LE)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))

	// stamp.nanosec (uint32 LE)
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))

	// frame_id CDR string
	buf = zenohsession.AppendCDRString(buf, frameID)

	// text CDR string
	buf = zenohsession.AppendCDRString(buf, text)

	return buf
}
