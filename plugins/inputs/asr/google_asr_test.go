package asr

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
	s := &transcriberStream{log: zap.NewNop(), provider: "google", language: "english", apiVersion: "v2"}

	require.Empty(t, googleParseMessage(s, ASRMessage{Type: "speech_start"}))
	require.True(t, s.speechStarted)

	require.Equal(t, "hello there world", googleParseMessage(s, ASRMessage{ASRReply: "hello there world"}))
	require.False(t, s.speechStarted)

	require.Empty(t, googleParseMessage(s, ASRMessage{ASRReply: "hi"}))
}
