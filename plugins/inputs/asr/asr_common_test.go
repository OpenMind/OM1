package asr

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestCJKRegex(t *testing.T) {
	require.True(t, cjkRegex.MatchString("你好"))
	require.True(t, cjkRegex.MatchString("こんにちは"))
	require.True(t, cjkRegex.MatchString("안녕"))
	require.False(t, cjkRegex.MatchString("hello"))
}

func TestSerializeASRText(t *testing.T) {
	buf := serializeASRText("hi")
	require.Equal(t, []byte{0x00, 0x01, 0x00, 0x00}, buf[:4])
	require.Contains(t, string(buf), "hi")
}

func TestResolveCooldown(t *testing.T) {
	require.Equal(t, time.Second, resolveCooldown(false, time.Second))
	require.Equal(t, time.Duration(0), resolveCooldown(false, 0))

	require.Equal(t, time.Duration(0), resolveCooldown(true, time.Second))
	require.Equal(t, time.Duration(0), resolveCooldown(true, 0))
}

func TestPushTranscriptCooldownSuppressesFollowup(t *testing.T) {
	a := newTestSensorCore()
	a.cooldown = time.Second

	// First transcript is accepted.
	a.pushTranscript("hello there world")
	got, ok := recvTranscript(t, a.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "hello there world", got)

	a.pushTranscript("today please")
	_, ok = recvTranscript(t, a.transcriptCh)
	require.False(t, ok, "follow-up within cooldown must be suppressed")
}

func TestPushTranscriptCooldownExpired(t *testing.T) {
	a := newTestSensorCore()
	a.cooldown = 10 * time.Millisecond

	a.pushTranscript("hello there world")
	got, ok := recvTranscript(t, a.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "hello there world", got)

	a.mu.Lock()
	a.lastDeliver = time.Now().Add(-time.Second)
	a.mu.Unlock()

	a.pushTranscript("a second utterance")
	got, ok = recvTranscript(t, a.transcriptCh)
	require.True(t, ok, "transcript after cooldown must be delivered")
	require.Equal(t, "a second utterance", got)
}

func TestPushTranscriptNoCooldownDeliversAll(t *testing.T) {
	a := newTestSensorCore()

	a.pushTranscript("hello there world")
	a.pushTranscript("today please")

	got, ok := recvTranscript(t, a.transcriptCh)
	require.True(t, ok)
	require.Equal(t, "hello there world", got)

	got, ok = recvTranscript(t, a.transcriptCh)
	require.True(t, ok, "with cooldown disabled every transcript is delivered")
	require.Equal(t, "today please", got)
}
