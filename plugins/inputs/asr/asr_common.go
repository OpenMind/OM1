package asr

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
	"github.com/openmind/om1/internal/providers/tts"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// cjkRegex matches CJK characters (Chinese, Japanese kana, Hangul), which use a
// shorter minimum-length threshold for accepting a transcript.
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
	Name               string
	Provider           string
	APIVersion         string
	WSURL              string
	Rate               int
	Language           string
	LanguageCode       string
	AltCodes           []string
	EnableTTSInterrupt bool
	ParseMessage       asrMessageParser
}

// asrCommon couples the sensor core with a single vendor stream.
type asrCommon struct {
	*asrSensorCore
	stream *transcriberStream
}

// asrSensorCore holds the parts shared by every ASR sensor: the transcript channel
// read by the cortex loop, the utterance buffer, lifecycle state, and the zenoh broadcast.
type asrSensorCore struct {
	name string
	log  *zap.Logger

	enableTTSInterrupt bool

	transcriptCh chan string

	messages []string

	mu          sync.Mutex
	stopped     bool
	captureDone chan struct{}

	zenohSession   zenohsession.Session
	zenohPublisher zenohsession.Publisher
}

// newSensorCore builds the sensor core, initializing the zenoh publisher if possible.
func newSensorCore(name string, enableTTSInterrupt bool) *asrSensorCore {
	log := logger.Get().Named(name)

	a := &asrSensorCore{
		name:               name,
		log:                log,
		enableTTSInterrupt: enableTTSInterrupt,
		transcriptCh:       make(chan string, 32),
	}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn("zenoh unavailable, ASR broadcast disabled", zap.Error(err))
		return a
	}
	pub, err := sess.DeclarePublisher(asrZenohTopic)
	if err != nil {
		sess.Close()
		log.Warn("failed to declare zenoh publisher, ASR broadcast disabled", zap.Error(err))
		return a
	}
	a.zenohSession = sess
	a.zenohPublisher = pub
	log.Info("zenoh publisher initialized", zap.String("topic", asrZenohTopic))

	return a
}

// pushTranscript accepts a transcript, pushing it onto the channel for the cortex loop.
func (a *asrSensorCore) pushTranscript(text string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.stopped {
		return
	}

	a.log.Info("transcript accepted", zap.String("text", text))

	if a.enableTTSInterrupt && tts.Speaking.Load() {
		tts.RequestInterrupt()
		a.log.Info("interrupting TTS due to detected speech")
	}

	select {
	case a.transcriptCh <- text:
	default:
		a.log.Warn("transcript buffer full, dropping", zap.String("text", text))
	}
}

// deliver is the default onTranscript handler for a single-provider stream: it
// ignores the provider label and forwards every accepted transcript.
func (a *asrSensorCore) deliver(_ string, text string) {
	a.pushTranscript(text)
}

// Poll returns the next accepted transcript, blocking until one is available or
// ctx is cancelled.
func (a *asrSensorCore) Poll(ctx context.Context) (any, error) {
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
func (a *asrSensorCore) pollLoop(ctx context.Context, out chan any) {
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
func (a *asrSensorCore) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
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
func (a *asrSensorCore) TriggersTick() bool { return true }

// FormattedLatestBuffer returns the buffered utterance as a Voice block, records
// it on the IO provider, broadcasts it over zenoh, and clears the buffer.
func (a *asrSensorCore) FormattedLatestBuffer() string {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.messages) == 0 {
		return ""
	}

	latest := a.messages[len(a.messages)-1]
	result := fmt.Sprintf("\nVoice: %q\n", latest)

	a.log.Info("flushing buffer", zap.String("text", latest))
	a.messages = nil

	providers.IO().AddInput("Voice", latest, time.Time{})

	if a.zenohPublisher != nil {
		payload := serializeASRText(latest)
		if err := a.zenohPublisher.Put(payload); err != nil {
			a.log.Warn("zenoh publish failed", zap.Error(err))
		} else {
			a.log.Info("published ASR to zenoh", zap.String("text", latest))
		}
	}

	return result
}

// markStopped flips the stopped flag exactly once. It returns whether this call
// was the first to stop the sensor and a snapshot of captureDone to wait on.
func (a *asrSensorCore) markStopped() (firstStop bool, captureDone chan struct{}) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.stopped {
		return false, nil
	}
	a.stopped = true
	return true, a.captureDone
}

// waitCapture waits up to 5s for the capture loop to finish.
func (a *asrSensorCore) waitCapture(captureDone chan struct{}) {
	if captureDone == nil {
		return
	}
	select {
	case <-captureDone:
	case <-time.After(5 * time.Second):
		a.log.Warn("capture loop did not stop within timeout")
	}
}

// closeZenoh drops the zenoh publisher and closes the session.
func (a *asrSensorCore) closeZenoh() {
	if a.zenohPublisher != nil {
		a.zenohPublisher.Drop()
		a.zenohPublisher = nil
		a.log.Info("zenoh publisher dropped")
	}
	if a.zenohSession != nil {
		a.zenohSession.Close()
		a.zenohSession = nil
		a.log.Info("zenoh session closed")
	}
}

// newASRCommon constructs an asrCommon from a vendor-resolved config, building the
// sensor core (with zenoh publisher) and the single websocket stream.
func newASRCommon(cfg asrCommonConfig) *asrCommon {
	agg := newSensorCore(cfg.Name, cfg.EnableTTSInterrupt)
	agg.log.Info("initializing",
		zap.String("provider", cfg.Provider),
		zap.String("language", cfg.Language),
		zap.String("language_code", cfg.LanguageCode),
		zap.String("api_version", cfg.APIVersion),
		zap.Int("rate", cfg.Rate),
		zap.Bool("enable_tts_interrupt", cfg.EnableTTSInterrupt),
	)

	stream := newTranscriberStream(cfg, agg.log, agg.deliver)

	return &asrCommon{asrSensorCore: agg, stream: stream}
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
	buf = append(buf, 0x00, 0x01, 0x00, 0x00) // CDR encapsulation header (little-endian)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))

	// stamp.nanosec (uint32 LE)
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))

	// frame_id CDR string
	buf = zenohsession.AppendCDRString(buf, frameID)

	// text CDR string
	buf = zenohsession.AppendCDRString(buf, text)

	return buf
}
