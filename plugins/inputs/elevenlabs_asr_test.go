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

func newTestElevenLabsCommon() *asrCommon {
	return &asrCommon{
		name:         "ElevenLabsASRInputTest",
		log:          zap.NewNop(),
		rate:         16000,
		model:        "elevenlabs",
		apiVersion:   elevenlabsAPIVersion,
		language:     "english",
		languageCode: "en",
		transcriptCh: make(chan string, 8),
		parseMessage: elevenlabsParseMessage,
	}
}

func committedMsg(t *testing.T, reply string) []byte {
	t.Helper()
	b, err := json.Marshal(ASRMessage{Type: "committed", ASRReply: reply})
	require.NoError(t, err)
	return b
}

func partialMsg(t *testing.T, reply string) []byte {
	t.Helper()
	b, err := json.Marshal(ASRMessage{Type: "partial", ASRReply: reply})
	require.NoError(t, err)
	return b
}

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
		{"two english words", "hello world", false},
		{"three english words", "hello there world", true},
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

	hLen := binary.BigEndian.Uint32(packet[0:4])
	require.Equal(t, len(packet), 4+int(hLen)+len(pcm))

	var header map[string]any
	require.NoError(t, json.Unmarshal(packet[4:4+hLen], &header))
	require.EqualValues(t, 16000, header["rate"])
	require.Equal(t, "zh", header["language_code"])
	require.Contains(t, header, "timestamp")
	require.NotContains(t, header, "alternative_language_codes",
		"ElevenLabs sends no alternative language codes; omitempty must drop the field")

	require.Equal(t, pcm, packet[4+hLen:])
}

func TestOnWSMessageCommittedDelivers(t *testing.T) {
	c := newTestElevenLabsCommon()
	c.speechStarted = true
	c.speechStartTime = time.Now().Add(-50 * time.Millisecond)

	c.onWSMessage(websocket.TextMessage, committedMsg(t, "hello there world"))

	got, ok := recvTranscript(t, c.transcriptCh)
	require.True(t, ok, "expected a transcript to be delivered")
	require.Equal(t, "hello there world", got)

	c.mu.Lock()
	require.False(t, c.speechStarted, "speech timing should reset after a transcript is accepted")
	c.mu.Unlock()
}

func TestOnWSMessagePartialMarksSpeechStart(t *testing.T) {
	c := newTestElevenLabsCommon()

	c.onWSMessage(websocket.TextMessage, partialMsg(t, "partial text"))

	c.mu.Lock()
	started := c.speechStarted
	c.mu.Unlock()
	require.True(t, started, "a partial message should mark the start of speech")

	_, ok := recvTranscript(t, c.transcriptCh)
	require.False(t, ok, "a partial message must not deliver a transcript")
}

func TestOnWSMessageRepeatedPartialsKeepFirstStart(t *testing.T) {
	c := newTestElevenLabsCommon()

	c.onWSMessage(websocket.TextMessage, partialMsg(t, "he"))
	c.mu.Lock()
	require.True(t, c.speechStarted)
	first := c.speechStartTime
	c.mu.Unlock()

	c.onWSMessage(websocket.TextMessage, partialMsg(t, "hello wor"))
	c.mu.Lock()
	require.True(t, c.speechStarted)
	require.Equal(t, first, c.speechStartTime, "start time should be set once per utterance")
	c.mu.Unlock()
}

func TestOnWSMessageCommittedResetsForNextUtterance(t *testing.T) {
	c := newTestElevenLabsCommon()
	c.speechStarted = true
	c.speechStartTime = time.Now().Add(-time.Second)

	c.onWSMessage(websocket.TextMessage, committedMsg(t, "hi"))
	_, ok := recvTranscript(t, c.transcriptCh)
	require.False(t, ok)
	c.mu.Lock()
	require.False(t, c.speechStarted, "a committed message must end the speech segment")
	c.mu.Unlock()

	c.onWSMessage(websocket.TextMessage, partialMsg(t, "how"))
	c.mu.Lock()
	require.True(t, c.speechStarted)
	require.WithinDuration(t, time.Now(), c.speechStartTime, time.Second)
	c.mu.Unlock()
}

func TestOnWSMessageIgnoresInvalidOrShort(t *testing.T) {
	cases := []struct {
		name    string
		msgType int
		data    []byte
	}{
		{"too short (single word)", websocket.TextMessage, committedMsg(t, "hi")},
		{"non-committed type", websocket.TextMessage, partialMsg(t, "hello world")},
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

	msg, err := c.RawToText(ctx, 123)
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = c.RawToText(ctx, "")
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = c.RawToText(ctx, "hello")
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "hello", msg.Message)

	_, err = c.RawToText(ctx, "world")
	require.NoError(t, err)

	c.mu.Lock()
	require.Equal(t, []string{"hello world"}, c.messages)
	c.mu.Unlock()
}

func TestFormattedLatestBuffer(t *testing.T) {
	c := newTestElevenLabsCommon()

	require.Equal(t, "", c.FormattedLatestBuffer())

	_, err := c.RawToText(context.Background(), "hello world")
	require.NoError(t, err)

	require.Equal(t, "\nVoice: \"hello world\"\n", c.FormattedLatestBuffer())

	require.Equal(t, "", c.FormattedLatestBuffer())
	c.mu.Lock()
	require.Empty(t, c.messages)
	c.mu.Unlock()
}
