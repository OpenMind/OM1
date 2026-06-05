package asr

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func newTestParallelSensor(window time.Duration) *ParallelASRSensor {
	return &ParallelASRSensor{
		asrSensorCore: newTestSensorCore(),
		dedupWindow:   window,
	}
}

func TestParallelDeliverFirstWins(t *testing.T) {
	s := newTestParallelSensor(time.Hour)

	// First provider to transcribe an utterance wins.
	s.deliver("riva", "hello there world")
	got, ok := recvTranscript(t, s.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "hello there world", got)

	// A different provider's transcript within the window is suppressed as a duplicate.
	s.deliver("google", "hello there world again")
	_, ok = recvTranscript(t, s.transcriptCh)
	require.False(t, ok, "the slower provider must be suppressed within the dedup window")

	// The winning provider is never suppressed; its next utterance passes through.
	s.deliver("riva", "second utterance here")
	got, ok = recvTranscript(t, s.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "second utterance here", got)
}

func TestParallelDeliverFailover(t *testing.T) {
	s := newTestParallelSensor(time.Hour)

	s.deliver("riva", "winner takes this")
	got, ok := recvTranscript(t, s.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "winner takes this", got)

	// Simulate the winner going silent past the window: another provider takes over.
	s.dmu.Lock()
	s.lastTime = time.Now().Add(-2 * time.Hour)
	s.dmu.Unlock()

	s.deliver("google", "failover transcript")
	got, ok = recvTranscript(t, s.transcriptCh)
	require.True(t, ok, "after the window expires a different provider should win")
	require.Equal(t, "failover transcript", got)

	s.dmu.Lock()
	require.Equal(t, "google", s.lastProvider)
	s.dmu.Unlock()
}

func TestParallelBuildStreamConfig(t *testing.T) {
	s := &ParallelASRSensor{cfg: ParallelASRConfig{Rate: 16000}}

	riva, err := s.buildStreamConfig(ParallelASRProviderConfig{Provider: "riva", BaseURL: "ws://localhost:6790"})
	require.NoError(t, err)
	require.Equal(t, "riva", riva.Provider)
	require.Equal(t, "ws://localhost:6790", riva.WSURL)

	_, err = s.buildStreamConfig(ParallelASRProviderConfig{Provider: "google"})
	require.Error(t, err, "google requires an api_key")

	_, err = s.buildStreamConfig(ParallelASRProviderConfig{Provider: "elevenlabs"})
	require.Error(t, err, "elevenlabs requires an api_key")

	_, err = s.buildStreamConfig(ParallelASRProviderConfig{Provider: "whisper"})
	require.Error(t, err, "unknown provider must be rejected")
}

func TestNewParallelASRValidation(t *testing.T) {
	_, err := NewParallelASR(map[string]any{"providers": []any{}})
	require.Error(t, err, "at least one provider is required")

	_, err = NewParallelASR(map[string]any{
		"source":    "carrier_pigeon",
		"providers": []any{map[string]any{"model": "riva"}},
	})
	require.Error(t, err, "invalid source must be rejected")
}
