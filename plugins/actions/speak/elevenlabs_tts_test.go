package speak

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

func TestRateFromFormat(t *testing.T) {
	require.Equal(t, 44100, rateFromFormat("pcm_44100"))
	require.Equal(t, 22050, rateFromFormat("mp3_44100_22050"), "takes the trailing numeric segment")
	require.Equal(t, providers.DefaultRate, rateFromFormat("pcm"), "no numeric suffix → default rate")
	require.Equal(t, providers.DefaultRate, rateFromFormat("pcm_abc"), "non-numeric suffix → default rate")
}

func newPeopleConnector(defaultVoice string, voiceIDs map[string]string) *ElevenLabsPeopleConnector {
	return &ElevenLabsPeopleConnector{
		ElevenLabsConnector: &ElevenLabsConnector{log: zap.NewNop()},
		defaultVoiceID:      defaultVoice,
		voiceIDs:            voiceIDs,
	}
}

func TestResolveVoiceIDDefaultWhenNoPerson(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.RemoveInput(facePresenceIOKey)
	t.Cleanup(func() { io.RemoveInput(facePresenceIOKey); io.ResetTickCounter() })

	c := newPeopleConnector("default-voice", map[string]string{"alice": "voice-a"})
	require.Equal(t, "default-voice", c.resolveVoiceID(), "no FacePresence input → default voice")
}

func TestResolveVoiceIDMatchesPerson(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.IncrementTick()
	io.AddInput(facePresenceIOKey, "In Camera View: 1 known (alice). Closest: alice.", time.Now())
	t.Cleanup(func() { io.RemoveInput(facePresenceIOKey); io.ResetTickCounter() })

	c := newPeopleConnector("default-voice", map[string]string{"alice": "voice-a"})
	require.Equal(t, "voice-a", c.resolveVoiceID(), "the closest known person's voice is selected")
}

func TestResolveVoiceIDUnknownPersonFallsBack(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.IncrementTick()
	io.AddInput(facePresenceIOKey, "Closest: unknown.", time.Now())
	t.Cleanup(func() { io.RemoveInput(facePresenceIOKey); io.ResetTickCounter() })

	c := newPeopleConnector("default-voice", map[string]string{"alice": "voice-a"})
	require.Equal(t, "default-voice", c.resolveVoiceID(), "an unknown closest person uses the default voice")
}
