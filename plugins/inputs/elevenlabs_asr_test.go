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

// newTestStream builds a transcriberStream whose accepted transcripts are pushed
// onto ch, for testing a vendor's parser and audio packaging in isolation.
func newTestStream(model, languageCode string, parser asrMessageParser, ch chan string) *transcriberStream {
	return &transcriberStream{
		name:         model + "Test",
		log:          zap.NewNop(),
		rate:         16000,
		model:        model,
		apiVersion:   elevenlabsAPIVersion,
		language:     "english",
		languageCode: languageCode,
		parseMessage: parser,
		onTranscript: func(_ string, text string) {
			select {
			case ch <- text:
			default:
			}
		},
	}
}

func newTestElevenLabsStream(ch chan string) *transcriberStream {
	return newTestStream("elevenlabs", "en", elevenlabsParseMessage, ch)
}

// newTestAggregator builds an asrAggregator with a no-op logger and no zenoh, for
// testing the transcript buffer independent of any vendor stream.
func newTestAggregator() *asrAggregator {
	return &asrAggregator{
		name:         "test",
		log:          zap.NewNop(),
		transcriptCh: make(chan string, 8),
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
	s := newTestElevenLabsStream(make(chan string, 1))
	s.rate = 16000
	s.languageCode = "zh"

	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	packet, err := s.packageAudio(pcm)
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
	ch := make(chan string, 1)
	s := newTestElevenLabsStream(ch)
	s.speechStarted = true
	s.speechStartTime = time.Now().Add(-50 * time.Millisecond)

	s.onWSMessage(websocket.TextMessage, committedMsg(t, "hello world"))

	got, ok := recvTranscript(t, ch)
	require.True(t, ok, "expected a transcript to be delivered")
	require.Equal(t, "hello world", got)

	s.mu.Lock()
	require.False(t, s.speechStarted, "speech timing should reset after a transcript is accepted")
	s.mu.Unlock()
}

func TestOnWSMessagePartialMarksSpeechStart(t *testing.T) {
	ch := make(chan string, 1)
	s := newTestElevenLabsStream(ch)

	s.onWSMessage(websocket.TextMessage, partialMsg(t, "partial text"))

	s.mu.Lock()
	started := s.speechStarted
	s.mu.Unlock()
	require.True(t, started, "a partial message should mark the start of speech")

	_, ok := recvTranscript(t, ch)
	require.False(t, ok, "a partial message must not deliver a transcript")
}

func TestOnWSMessageRepeatedPartialsKeepFirstStart(t *testing.T) {
	ch := make(chan string, 1)
	s := newTestElevenLabsStream(ch)

	s.onWSMessage(websocket.TextMessage, partialMsg(t, "he"))
	s.mu.Lock()
	require.True(t, s.speechStarted)
	first := s.speechStartTime
	s.mu.Unlock()

	s.onWSMessage(websocket.TextMessage, partialMsg(t, "hello wor"))
	s.mu.Lock()
	require.True(t, s.speechStarted)
	require.Equal(t, first, s.speechStartTime, "start time should be set once per utterance")
	s.mu.Unlock()
}

func TestOnWSMessageCommittedResetsForNextUtterance(t *testing.T) {
	ch := make(chan string, 1)
	s := newTestElevenLabsStream(ch)
	s.speechStarted = true
	s.speechStartTime = time.Now().Add(-time.Second)

	s.onWSMessage(websocket.TextMessage, committedMsg(t, "hi"))
	_, ok := recvTranscript(t, ch)
	require.False(t, ok)
	s.mu.Lock()
	require.False(t, s.speechStarted, "a committed message must end the speech segment")
	s.mu.Unlock()

	s.onWSMessage(websocket.TextMessage, partialMsg(t, "how"))
	s.mu.Lock()
	require.True(t, s.speechStarted)
	require.WithinDuration(t, time.Now(), s.speechStartTime, time.Second)
	s.mu.Unlock()
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
			ch := make(chan string, 1)
			s := newTestElevenLabsStream(ch)
			s.onWSMessage(tc.msgType, tc.data)
			_, ok := recvTranscript(t, ch)
			require.False(t, ok, "no transcript should be delivered for %q", tc.name)
		})
	}
}

func TestAggregatorStoppedIgnores(t *testing.T) {
	a := newTestAggregator()
	a.stopped = true

	a.pushTranscript("hello world")

	select {
	case <-a.transcriptCh:
		t.Fatal("a stopped aggregator must drop transcripts")
	default:
	}
}

func TestRawToText(t *testing.T) {
	a := newTestAggregator()
	ctx := context.Background()

	msg, err := a.RawToText(ctx, 123)
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = a.RawToText(ctx, "")
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = a.RawToText(ctx, "hello")
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "hello", msg.Message)

	_, err = a.RawToText(ctx, "world")
	require.NoError(t, err)

	a.mu.Lock()
	require.Equal(t, []string{"hello world"}, a.messages)
	a.mu.Unlock()
}

func TestFormattedLatestBuffer(t *testing.T) {
	a := newTestAggregator()

	require.Equal(t, "", a.FormattedLatestBuffer())

	_, err := a.RawToText(context.Background(), "hello world")
	require.NoError(t, err)

	require.Equal(t, "\nVoice: \"hello world\"\n", a.FormattedLatestBuffer())

	require.Equal(t, "", a.FormattedLatestBuffer())
	a.mu.Lock()
	require.Empty(t, a.messages)
	a.mu.Unlock()
}
