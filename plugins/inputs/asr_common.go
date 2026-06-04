package inputs

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// cjkRegex matches CJK characters (Chinese, Japanese kana, Hangul) used to decide
// the minimum-length threshold for accepting a transcript.
var cjkRegex = regexp.MustCompile(`[\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}]`)

const asrZenohTopic = "om/asr/text"

// acceptASRTranscript reports whether a transcript is long enough to keep.
func acceptASRTranscript(text string) bool {
	if cjkRegex.MatchString(text) {
		return utf8.RuneCountInString(text) > 2
	}
	return len(strings.Fields(text)) > 2
}

// ASRMessage is a transcript/status message received from the ASR websocket.
type ASRMessage struct {
	Type     string `json:"type"`
	ASRReply string `json:"asr_reply"`
	Time     int64  `json:"time"`
}

// asrMessageParser handles one decoded ASR message under the stream's mu and
// returns the transcript to deliver ("" if none).
type asrMessageParser func(s *transcriberStream, msg ASRMessage) string

// asrCommonConfig holds the vendor-resolved configuration for a transcriberStream.
type asrCommonConfig struct {
	Name               string // log prefix, e.g. "GoogleASRInput"
	Model              string // metric label, e.g. "google" / "elevenlabs"
	APIVersion         string // metric label, e.g. "v1" / "v2"
	WSURL              string // fully-built websocket endpoint
	Rate               int
	Language           string // friendly name, used as a metric label
	LanguageCode       string
	AltCodes           []string
	EnableTTSInterrupt bool
	ParseMessage       asrMessageParser
}

// asrCommon couples the aggregator with a single vendor stream.
type asrCommon struct {
	*asrAggregator
	stream *transcriberStream
}

// asrAggregator owns the ASR sensor parts: the transcript channel read by the cortex loop,
// the current-utterance buffer, lifecycle state, and the zenoh broadcast
type asrAggregator struct {
	name string
	log  *zap.Logger

	transcriptCh chan string

	messages []string

	mu          sync.Mutex
	stopped     bool
	captureDone chan struct{}

	zenohSession   *zenohsession.Session
	zenohPublisher *zenohsession.Publisher
}

// newAggregator constructs an asrAggregator, opening the (optional) zenoh publisher
// used to broadcast transcripts on om/asr/text.
func newAggregator(name string) *asrAggregator {
	log := logger.Get()

	a := &asrAggregator{
		name:         name,
		log:          log,
		transcriptCh: make(chan string, 32),
	}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn(name+": zenoh unavailable, ASR broadcast disabled", zap.Error(err))
		return a
	}
	pub, err := sess.DeclarePublisher(asrZenohTopic)
	if err != nil {
		sess.Close()
		log.Warn(name+": failed to declare zenoh publisher, ASR broadcast disabled", zap.Error(err))
		return a
	}
	a.zenohSession = sess
	a.zenohPublisher = pub
	log.Info(name+": zenoh publisher initialized", zap.String("topic", asrZenohTopic))

	return a
}

// pushTranscript delivers an accepted transcript to the cortex loop unless the
// sensor has stopped.
func (a *asrAggregator) pushTranscript(text string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.stopped {
		return
	}

	a.log.Info(a.name+": transcript accepted", zap.String("text", text))

	select {
	case a.transcriptCh <- text:
	default:
		a.log.Warn(a.name+": transcript buffer full, dropping", zap.String("text", text))
	}
}

// deliver is the default onTranscript handler for a single-provider stream: it
// ignores the provider label and forwards every accepted transcript.
func (a *asrAggregator) deliver(_ string, text string) {
	a.pushTranscript(text)
}

// Poll returns the next accepted transcript, blocking until one is available or
// ctx is cancelled.
func (a *asrAggregator) Poll(ctx context.Context) (any, error) {
	select {
	case text, ok := <-a.transcriptCh:
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
func (a *asrAggregator) pollLoop(ctx context.Context, out chan any) {
	for {
		raw, err := a.Poll(ctx)
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
func (a *asrAggregator) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.messages) == 0 {
		a.messages = append(a.messages, text)
	} else {
		a.messages[len(a.messages)-1] = a.messages[len(a.messages)-1] + " " + text
	}

	return inputs.NewMessage(text), nil
}

// TriggersTick reports that ASR input wakes the cortex loop.
func (a *asrAggregator) TriggersTick() bool { return true }

// FormattedLatestBuffer returns the buffered utterance as a Voice block, records
// it on the IO provider, broadcasts it over zenoh, and clears the buffer.
func (a *asrAggregator) FormattedLatestBuffer() string {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.messages) == 0 {
		return ""
	}

	latest := a.messages[len(a.messages)-1]
	result := fmt.Sprintf("\nVoice: %q\n", latest)

	a.log.Info(a.name+": flushing buffer", zap.String("text", latest))
	a.messages = nil

	providers.IO().AddInput("Voice", latest, time.Time{})

	if a.zenohPublisher != nil {
		payload := serializeASRText(latest)
		if err := a.zenohPublisher.Put(payload); err != nil {
			a.log.Warn(a.name+": zenoh publish failed", zap.Error(err))
		} else {
			a.log.Info(a.name+": published ASR to zenoh", zap.String("text", latest))
		}
	}

	return result
}

// markStopped flips the stopped flag exactly once. It returns whether this call
// was the first to stop the sensor and a snapshot of captureDone to wait on.
func (a *asrAggregator) markStopped() (firstStop bool, captureDone chan struct{}) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.stopped {
		return false, nil
	}
	a.stopped = true
	return true, a.captureDone
}

// waitCapture waits up to 5s for the capture loop to finish.
func (a *asrAggregator) waitCapture(captureDone chan struct{}) {
	if captureDone == nil {
		return
	}
	select {
	case <-captureDone:
	case <-time.After(5 * time.Second):
		a.log.Warn(a.name + ": capture loop did not stop within timeout")
	}
}

// closeZenoh drops the zenoh publisher and closes the session.
func (a *asrAggregator) closeZenoh() {
	if a.zenohPublisher != nil {
		a.zenohPublisher.Drop()
		a.zenohPublisher = nil
		a.log.Info(a.name + ": zenoh publisher dropped")
	}
	if a.zenohSession != nil {
		a.zenohSession.Close()
		a.zenohSession = nil
		a.log.Info(a.name + ": zenoh session closed")
	}
}

// newASRCommon constructs an asrCommon from a vendor-resolved config, building the
// aggregator (with zenoh publisher) and the single websocket stream.
func newASRCommon(cfg asrCommonConfig) *asrCommon {
	agg := newAggregator(cfg.Name)
	agg.log.Info(cfg.Name+": initializing",
		zap.String("model", cfg.Model),
		zap.String("language", cfg.Language),
		zap.String("language_code", cfg.LanguageCode),
		zap.String("api_version", cfg.APIVersion),
		zap.Int("rate", cfg.Rate),
	)

	stream := newTranscriberStream(cfg, agg.log, agg.deliver)

	return &asrCommon{asrAggregator: agg, stream: stream}
}

// connect dials the single stream's ASR websocket.
func (c *asrCommon) connect() error { return c.stream.connect() }

// sendChunk forwards a PCM chunk to the single stream's websocket.
func (c *asrCommon) sendChunk(pcm []byte) { c.stream.sendChunk(pcm) }

// statsLoop logs the single stream's send statistics until ctx is cancelled.
func (c *asrCommon) statsLoop(ctx context.Context) { c.stream.statsLoop(ctx) }

// closeWS closes the single stream's websocket client.
func (c *asrCommon) closeWS() { c.stream.closeWS() }

// serializeASRText encodes an ASRText message in CDR little-endian format.
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
