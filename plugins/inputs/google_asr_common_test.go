package inputs

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestLanguageCodeMap(t *testing.T) {
	require.Equal(t, "en-US", languageCodeMap["english"])
	require.Equal(t, "cmn-Hans-CN", languageCodeMap["chinese"])
	_, ok := languageCodeMap["klingon"]
	require.False(t, ok)
}

func TestCJKRegex(t *testing.T) {
	require.True(t, cjkRegex.MatchString("你好"))
	require.True(t, cjkRegex.MatchString("こんにちは"))
	require.True(t, cjkRegex.MatchString("안녕"))
	require.False(t, cjkRegex.MatchString("hello"))
}

func TestAcceptTranscript(t *testing.T) {
	c := &googleASRCommon{}
	require.True(t, c.acceptTranscript("hello world"), "multi-word latin transcript accepted")
	require.False(t, c.acceptTranscript("hello"), "single latin word rejected")
	require.True(t, c.acceptTranscript("你好吗"), "CJK with >2 runes accepted")
	require.False(t, c.acceptTranscript("你好"), "CJK with 2 runes rejected")
}

func TestASRRawToTextAccumulates(t *testing.T) {
	c := &googleASRCommon{log: zap.NewNop()}

	msg, err := c.RawToText(context.Background(), "hello")
	require.NoError(t, err)
	require.Equal(t, "hello", msg.Message)

	_, err = c.RawToText(context.Background(), "world")
	require.NoError(t, err)
	require.Equal(t, []string{"hello world"}, c.messages, "consecutive transcripts join into one utterance")

	empty, err := c.RawToText(context.Background(), 123)
	require.NoError(t, err)
	require.Nil(t, empty, "non-string raw input is ignored")
}

func TestASRFormattedLatestBuffer(t *testing.T) {
	c := &googleASRCommon{log: zap.NewNop(), messages: []string{"good morning"}}
	out := c.FormattedLatestBuffer()
	require.Equal(t, "\nVoice: \"good morning\"\n", out)
	require.Empty(t, c.messages, "buffer is cleared after flushing")

	require.Equal(t, "", c.FormattedLatestBuffer(), "empty buffer yields empty string")
}

func TestPackageAudio(t *testing.T) {
	c := &googleASRCommon{rate: 16000, languageCode: "en-US"}
	pcm := []byte{1, 2, 3, 4}
	packet, err := c.packageAudio(pcm)
	require.NoError(t, err)

	hLen := binary.BigEndian.Uint32(packet[0:4])
	require.Equal(t, len(packet), 4+int(hLen)+len(pcm), "packet = 4-byte length prefix + header + pcm")

	var meta AudioMetadata
	require.NoError(t, json.Unmarshal(packet[4:4+hLen], &meta))
	require.Equal(t, 16000, meta.Rate)
	require.Equal(t, "en-US", meta.LanguageCode)

	require.Equal(t, pcm, packet[4+hLen:], "the pcm payload is appended verbatim")
}

func TestSerializeASRText(t *testing.T) {
	buf := serializeASRText("hi")
	require.Equal(t, []byte{0x00, 0x01, 0x00, 0x00}, buf[:4], "CDR encapsulation header")
	require.Contains(t, string(buf), "hi", "the transcript text is embedded")
}
