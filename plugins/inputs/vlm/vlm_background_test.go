package vlm

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	video "github.com/openmind/om1/internal/providers/vlm"
)

func TestVLMBackgroundRegisteredAndDefaults(t *testing.T) {
	s, err := inputs.Load("VLMBackground", map[string]any{})
	require.NoError(t, err)
	require.NotNil(t, s)

	reader, ok := s.(*vlmBackgroundSensor)
	require.True(t, ok)
	assert.Equal(t, time.Duration(vlmBackgroundDefaultPollSec*float64(time.Second)), reader.period)
	assert.Equal(t, time.Duration(0), reader.maxAge, "max age disabled by default")
	s.Stop()
}

func TestVLMBackgroundReadsLatestDescription(t *testing.T) {
	video.LatestDescription().Set("a person waving", time.Now())

	s := &vlmBackgroundSensor{log: zap.NewNop()}

	text, _, ok := s.read()
	require.True(t, ok)
	assert.Equal(t, "a person waving", text)

	raw, err := s.Poll(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "a person waving", raw)
}

func TestVLMBackgroundMaxAgeFiltersStale(t *testing.T) {
	s := &vlmBackgroundSensor{log: zap.NewNop(), maxAge: time.Second}
	video.LatestDescription().Set("a stale scene", time.Now().Add(-time.Minute))

	_, _, ok := s.read()
	assert.False(t, ok, "stale description should be filtered out")
}

func TestVLMBackgroundListenDedupesByTimestamp(t *testing.T) {
	ts := time.Now()
	video.LatestDescription().Set("scene one", ts)

	s := &vlmBackgroundSensor{log: zap.NewNop(), period: 10 * time.Millisecond}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, err := s.Listen(ctx)
	require.NoError(t, err)

	count := func() int {
		s.mu.Lock()
		defer s.mu.Unlock()
		return len(s.messages)
	}

	// The current description is emitted exactly once...
	require.Eventually(t, func() bool { return count() == 1 }, time.Second, 5*time.Millisecond)
	time.Sleep(40 * time.Millisecond)
	assert.Equal(t, 1, count(), "same-timestamp description is not re-emitted")

	// ...and a newer description is picked up.
	video.LatestDescription().Set("scene two", ts.Add(time.Second))
	require.Eventually(t, func() bool { return count() == 2 }, time.Second, 5*time.Millisecond)
}

func TestVLMBackgroundFormatAndClear(t *testing.T) {
	s := &vlmBackgroundSensor{log: zap.NewNop()}

	_, err := s.RawToText(context.Background(), "hello scene")
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	assert.Contains(t, out, vlmDescriptor)
	assert.Contains(t, out, "hello scene")
	assert.Equal(t, "", s.FormattedLatestBuffer(), "buffer cleared after read")
}

func TestVLMBackgroundRawToTextIgnoresNonStrings(t *testing.T) {
	s := &vlmBackgroundSensor{log: zap.NewNop()}
	msg, err := s.RawToText(context.Background(), 123)
	require.NoError(t, err)
	assert.Nil(t, msg)
	assert.Equal(t, "", s.FormattedLatestBuffer())
}

func TestVLMBackgroundBoundsHistory(t *testing.T) {
	s := &vlmBackgroundSensor{log: zap.NewNop()}
	for i := 0; i < vlmMaxMessages+5; i++ {
		_, _ = s.RawToText(context.Background(), "msg")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	assert.Len(t, s.messages, vlmMaxMessages)
}
