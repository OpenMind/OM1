package inputs

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestGoogleLanguageCodeMap(t *testing.T) {
	require.Equal(t, "en-US", googleLanguageCodeMap["english"])
	require.Equal(t, "cmn-Hans-CN", googleLanguageCodeMap["chinese"])
	_, ok := googleLanguageCodeMap["klingon"]
	require.False(t, ok)
}

func TestGoogleParseMessage(t *testing.T) {
	c := &asrCommon{log: zap.NewNop(), model: "google", language: "english", apiVersion: "v2"}

	require.Empty(t, googleParseMessage(c, ASRMessage{Type: "speech_start"}))
	require.True(t, c.speechStarted)

	require.Equal(t, "hello there world", googleParseMessage(c, ASRMessage{ASRReply: "hello there world"}))
	require.False(t, c.speechStarted)

	require.Empty(t, googleParseMessage(c, ASRMessage{ASRReply: "hi"}))
}
