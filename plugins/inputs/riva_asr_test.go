package inputs

import (
	"encoding/binary"
	"encoding/json"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func newTestRivaCommon() *asrCommon {
	return &asrCommon{
		name:         "RivaASRInputTest",
		log:          zap.NewNop(),
		rate:         48000,
		model:        "riva",
		apiVersion:   rivaAPIVersion,
		language:     "english",
		languageCode: "en-US",
		transcriptCh: make(chan string, 8),
		parseMessage: rivaParseMessage,
	}
}

func rivaReplyMsg(t *testing.T, reply string) []byte {
	t.Helper()
	b, err := json.Marshal(ASRMessage{ASRReply: reply})
	require.NoError(t, err)
	return b
}

func TestAcceptRivaTranscript(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
	}{
		{"empty", "", false},
		{"single word", "hello", false},
		{"two words", "hello world", false},
		{"three words", "hello there world", true},
		{"leading/trailing spaces two words", "   hello world   ", false},
		{"four words", "how are you doing", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, acceptRivaTranscript(tc.text))
		})
	}
}

func TestRivaLanguageCodeMap(t *testing.T) {
	require.Equal(t, "en-US", rivaLanguageCodeMap["english"])
	require.Equal(t, "cmn-Hans-CN", rivaLanguageCodeMap["chinese"])

	_, ok := rivaLanguageCodeMap["klingon"]
	require.False(t, ok, "unsupported language must be absent so the caller falls back to en-US")

	for name, code := range rivaLanguageCodeMap {
		require.NotEmpty(t, code, "language %q maps to an empty code", name)
	}
}

func TestRivaParseMessage(t *testing.T) {
	c := newTestRivaCommon()

	require.Equal(t, "hello there world", rivaParseMessage(c, ASRMessage{ASRReply: "hello there world"}))
	require.Empty(t, rivaParseMessage(c, ASRMessage{ASRReply: "hello world"}), "two words must be rejected")
	require.Empty(t, rivaParseMessage(c, ASRMessage{ASRReply: ""}), "empty reply must be rejected")
}

func TestRivaPackageAudioHeader(t *testing.T) {
	c := newTestRivaCommon()
	c.rate = 48000
	c.languageCode = "en-US"

	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	packet, err := c.packageAudio(pcm)
	require.NoError(t, err)

	hLen := binary.BigEndian.Uint32(packet[0:4])
	require.Equal(t, len(packet), 4+int(hLen)+len(pcm))

	var header map[string]any
	require.NoError(t, json.Unmarshal(packet[4:4+hLen], &header))
	require.EqualValues(t, 48000, header["rate"])
	require.Equal(t, "en-US", header["language_code"])
	require.Contains(t, header, "timestamp")
	require.NotContains(t, header, "alternative_language_codes",
		"Riva sends no alternative language codes; omitempty must drop the field")

	require.Equal(t, pcm, packet[4+hLen:])
}

func TestRivaOnWSMessageDelivers(t *testing.T) {
	c := newTestRivaCommon()

	c.onWSMessage(websocket.TextMessage, rivaReplyMsg(t, "hello there world"))

	got, ok := recvTranscript(t, c.transcriptCh)
	require.True(t, ok, "expected a transcript to be delivered")
	require.Equal(t, "hello there world", got)
}

func TestRivaOnWSMessageIgnoresInvalidOrShort(t *testing.T) {
	cases := []struct {
		name    string
		msgType int
		data    []byte
	}{
		{"too short (two words)", websocket.TextMessage, rivaReplyMsg(t, "hello world")},
		{"empty reply", websocket.TextMessage, rivaReplyMsg(t, "")},
		{"invalid json", websocket.TextMessage, []byte("{not json")},
		{"binary message", websocket.BinaryMessage, rivaReplyMsg(t, "hello there world")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestRivaCommon()
			c.onWSMessage(tc.msgType, tc.data)
			_, ok := recvTranscript(t, c.transcriptCh)
			require.False(t, ok, "no transcript should be delivered for %q", tc.name)
		})
	}
}

func TestRivaOnWSMessageStoppedIgnores(t *testing.T) {
	c := newTestRivaCommon()
	c.stopped = true

	c.onWSMessage(websocket.TextMessage, rivaReplyMsg(t, "hello there world"))

	_, ok := recvTranscript(t, c.transcriptCh)
	require.False(t, ok, "a stopped sensor must ignore incoming messages")
}
