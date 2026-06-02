package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/ws"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

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

// acceptASRTranscript reports whether a transcript is long enough to keep.
func acceptASRTranscript(text string) bool {
	if cjkRegex.MatchString(text) {
		return utf8.RuneCountInString(text) > 2
	}
	return len(strings.Fields(text)) > 1
}

// elevenlabsASRCommonConfig holds the configuration parameters for elevenlabsASRCommon.
type elevenlabsASRCommonConfig struct {
	APIKey   string
	BaseURL  string
	Rate     int
	Language string
}

// elevenlabsASRCommon encapsulates the shared logic for ElevenLabs ASR sensors.
type elevenlabsASRCommon struct {
	name         string // log prefix, e.g. "ElevenLabsASRInput"
	log          *zap.Logger
	rate         int
	language     string // friendly name, used as a metric label
	languageCode string

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

// NewElevenLabsASRCommon constructs an elevenlabsASRCommon with the given configuration.
func NewElevenLabsASRCommon(name string, cfg elevenlabsASRCommonConfig) *elevenlabsASRCommon {
	language := strings.TrimSpace(strings.ToLower(cfg.Language))
	if language == "" {
		language = "auto"
	}

	log := logger.Get()

	languageCode, ok := elevenlabsLanguageCodeMap[language]
	if !ok {
		log.Error(name+": unsupported language, defaulting to auto",
			zap.String("language", language))
		language = "auto"
		languageCode = "auto"
	}

	wsURL := cfg.BaseURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://api.openmind.com/api/core/elevenlabs/asr?api_key=%s", cfg.APIKey)
	}

	log.Info(name+": initializing",
		zap.String("language", language),
		zap.String("language_code", languageCode),
		zap.Int("rate", cfg.Rate),
	)

	c := &elevenlabsASRCommon{
		name:         name,
		log:          log,
		rate:         cfg.Rate,
		language:     language,
		languageCode: languageCode,
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
func (c *elevenlabsASRCommon) Poll(ctx context.Context) (any, error) {
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
func (c *elevenlabsASRCommon) pollLoop(ctx context.Context, out chan any) {
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
func (c *elevenlabsASRCommon) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
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
func (c *elevenlabsASRCommon) TriggersTick() bool { return true }

// FormattedLatestBuffer returns the buffered utterance as a Voice block, records
// it on the IO provider, broadcasts it over zenoh, and clears the buffer.
func (c *elevenlabsASRCommon) FormattedLatestBuffer() string {
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
func (c *elevenlabsASRCommon) packageAudio(pcm []byte) ([]byte, error) {
	meta := AudioMetadata{
		Rate:         c.rate,
		LanguageCode: c.languageCode,
		Timestamp:    time.Now().UnixMilli(),
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
func (c *elevenlabsASRCommon) sendChunk(pcm []byte) {
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

// onWSMessage parses ASR websocket messages, records latency metrics, and forwards
// accepted transcripts to the main loop.
func (c *elevenlabsASRCommon) onWSMessage(msgType int, data []byte) {
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

	if msg.Type == "partial" {
		if !c.speechStarted {
			c.speechStartTime = time.Now()
			c.speechStarted = true
		}
		return
	}

	if msg.Type != "committed" {
		return
	}

	speechStarted := c.speechStarted
	speechStartTime := c.speechStartTime
	c.speechStarted = false

	if msg.ASRReply == "" || !acceptASRTranscript(msg.ASRReply) {
		return
	}

	var latency time.Duration
	if speechStarted {
		latency = time.Since(speechStartTime)
		seconds := latency.Seconds()
		metrics.ASRLatency.WithLabelValues("elevenlabs", c.language, elevenlabsAPIVersion).Observe(seconds)
		metrics.ASRLatencyLast.WithLabelValues("elevenlabs", c.language, elevenlabsAPIVersion).Set(seconds)
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

// statsLoop logs send statistics every 15 seconds until ctx is cancelled.
func (c *elevenlabsASRCommon) statsLoop(ctx context.Context) {
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
func (c *elevenlabsASRCommon) PrintStatistics() {
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
func (c *elevenlabsASRCommon) markStopped() (firstStop bool, captureDone chan struct{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.stopped {
		return false, nil
	}
	c.stopped = true
	return true, c.captureDone
}

// waitCapture waits up to 5s for the capture loop to finish.
func (c *elevenlabsASRCommon) waitCapture(captureDone chan struct{}) {
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
func (c *elevenlabsASRCommon) closeWS() {
	if c.wsClient != nil {
		c.wsClient.Close()
	}
}

// closeZenoh drops the zenoh publisher and closes the session.
func (c *elevenlabsASRCommon) closeZenoh() {
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
