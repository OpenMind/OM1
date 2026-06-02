package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// newTestElevenLabsCommon builds an elevenlabsASRCommon without touching the
// network: no websocket dial, no zenoh session, no PortAudio. This lets us
// exercise the message-handling and buffering logic in isolation.
func newTestElevenLabsCommon() *elevenlabsASRCommon {
	return &elevenlabsASRCommon{
		name:         "ElevenLabsASRInputTest",
		log:          zap.NewNop(),
		rate:         16000,
		language:     "english",
		languageCode: "en",
		transcriptCh: make(chan string, 8),
	}
}

// committedMsg encodes an ElevenLabs "committed" transcript message.
func committedMsg(t *testing.T, reply string) []byte {
	t.Helper()
	b, err := json.Marshal(ASRMessage{Type: "committed", ASRReply: reply})
	require.NoError(t, err)
	return b
}

// recvTranscript reads one transcript from the channel, or reports that none is
// queued. onWSMessage delivers synchronously before returning, so by the time it
// returns the transcript is either already buffered or was never sent — a
// non-blocking read is therefore deterministic.
func recvTranscript(t *testing.T, ch chan string) (string, bool) {
	t.Helper()
	select {
	case s := <-ch:
		return s, true
	default:
		return "", false
	}
}

func TestAcceptASRTranscript(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
	}{
		{"empty", "", false},
		{"single english word", "hello", false},
		{"two english words", "hello world", true},
		{"leading/trailing spaces single word", "   hello   ", false},
		{"single CJK char", "好", false},
		{"two CJK chars", "你好", false},
		{"three CJK chars", "你好吗", true},
		{"japanese kana phrase", "こんにちは", true},
		{"korean phrase", "안녕하세요", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, acceptASRTranscript(tc.text))
		})
	}
}

func TestElevenLabsLanguageCodeMap(t *testing.T) {
	// Spot-check the friendly-name -> short-code mapping that distinguishes
	// ElevenLabs from the Google ASR (which uses BCP-47 codes).
	require.Equal(t, "auto", elevenlabsLanguageCodeMap["auto"])
	require.Equal(t, "en", elevenlabsLanguageCodeMap["english"])
	require.Equal(t, "zh", elevenlabsLanguageCodeMap["chinese"])
	require.Equal(t, "ja", elevenlabsLanguageCodeMap["japanese"])

	_, ok := elevenlabsLanguageCodeMap["klingon"]
	require.False(t, ok, "unsupported language must be absent so the caller falls back to auto")

	for name, code := range elevenlabsLanguageCodeMap {
		require.NotEmpty(t, code, "language %q maps to an empty code", name)
	}
}

func TestPackageAudio(t *testing.T) {
	c := newTestElevenLabsCommon()
	c.rate = 16000
	c.languageCode = "zh"

	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	packet, err := c.packageAudio(pcm)
	require.NoError(t, err)
	require.Greater(t, len(packet), 4+len(pcm))

	// First 4 bytes: big-endian header length.
	hLen := binary.BigEndian.Uint32(packet[0:4])
	require.Equal(t, len(packet), 4+int(hLen)+len(pcm))

	// Header is JSON with rate, language_code, timestamp; no alternative codes.
	var header map[string]any
	require.NoError(t, json.Unmarshal(packet[4:4+hLen], &header))
	require.EqualValues(t, 16000, header["rate"])
	require.Equal(t, "zh", header["language_code"])
	require.Contains(t, header, "timestamp")
	require.NotContains(t, header, "alternative_language_codes",
		"ElevenLabs sends no alternative language codes; omitempty must drop the field")

	// PCM payload is appended verbatim after the header.
	require.Equal(t, pcm, packet[4+hLen:])
}

func TestOnWSMessageCommittedDelivers(t *testing.T) {
	c := newTestElevenLabsCommon()
	c.speechStarted = true
	c.speechStartTime = time.Now().Add(-50 * time.Millisecond)

	c.onWSMessage(websocket.TextMessage, committedMsg(t, "hello world"))

	got, ok := recvTranscript(t, c.transcriptCh)
	require.True(t, ok, "expected a transcript to be delivered")
	require.Equal(t, "hello world", got)

	c.mu.Lock()
	require.False(t, c.speechStarted, "speech timing should reset after a transcript is accepted")
	c.mu.Unlock()
}

func TestOnWSMessagePartialMarksSpeechStart(t *testing.T) {
	c := newTestElevenLabsCommon()

	b, err := json.Marshal(ASRMessage{Type: "partial", ASRReply: "partial text"})
	require.NoError(t, err)
	c.onWSMessage(websocket.TextMessage, b)

	c.mu.Lock()
	started := c.speechStarted
	c.mu.Unlock()
	require.True(t, started, "a partial message should mark the start of speech")

	_, ok := recvTranscript(t, c.transcriptCh)
	require.False(t, ok, "a partial message must not deliver a transcript")
}

func TestOnWSMessageIgnoresInvalidOrShort(t *testing.T) {
	cases := []struct {
		name    string
		msgType int
		data    []byte
	}{
		{"too short (single word)", websocket.TextMessage, committedMsg(t, "hi")},
		{"non-committed type", websocket.TextMessage, func() []byte {
			b, _ := json.Marshal(ASRMessage{Type: "partial", ASRReply: "hello world"})
			return b
		}()},
		{"empty reply", websocket.TextMessage, committedMsg(t, "")},
		{"invalid json", websocket.TextMessage, []byte("{not json")},
		{"binary message", websocket.BinaryMessage, committedMsg(t, "hello world")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestElevenLabsCommon()
			c.onWSMessage(tc.msgType, tc.data)
			_, ok := recvTranscript(t, c.transcriptCh)
			require.False(t, ok, "no transcript should be delivered for %q", tc.name)
		})
	}
}

func TestOnWSMessageStoppedIgnores(t *testing.T) {
	c := newTestElevenLabsCommon()
	c.stopped = true

	c.onWSMessage(websocket.TextMessage, committedMsg(t, "hello world"))

	_, ok := recvTranscript(t, c.transcriptCh)
	require.False(t, ok, "a stopped sensor must ignore incoming messages")
}

func TestRawToText(t *testing.T) {
	c := newTestElevenLabsCommon()
	ctx := context.Background()

	// Non-string and empty inputs are dropped.
	msg, err := c.RawToText(ctx, 123)
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = c.RawToText(ctx, "")
	require.NoError(t, err)
	require.Nil(t, msg)

	// First transcript starts the utterance.
	msg, err = c.RawToText(ctx, "hello")
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "hello", msg.Message)

	// Subsequent transcripts are space-joined into the same utterance.
	_, err = c.RawToText(ctx, "world")
	require.NoError(t, err)

	c.mu.Lock()
	require.Equal(t, []string{"hello world"}, c.messages)
	c.mu.Unlock()
}

func TestFormattedLatestBuffer(t *testing.T) {
	c := newTestElevenLabsCommon()

	// Empty buffer yields an empty string.
	require.Equal(t, "", c.FormattedLatestBuffer())

	_, err := c.RawToText(context.Background(), "hello world")
	require.NoError(t, err)

	require.Equal(t, "\nVoice: \"hello world\"\n", c.FormattedLatestBuffer())

	// Flushing clears the buffer, so a second call is empty again.
	require.Equal(t, "", c.FormattedLatestBuffer())
	c.mu.Lock()
	require.Empty(t, c.messages)
	c.mu.Unlock()
}
