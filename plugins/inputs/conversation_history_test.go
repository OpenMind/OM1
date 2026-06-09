package inputs

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/providers"
)

func newConvHistory(t *testing.T, cfg map[string]any) *ConversationHistorySensor {
	t.Helper()
	s, err := NewConversationHistory(cfg)
	require.NoError(t, err)
	return s.(*ConversationHistorySensor)
}

func TestConversationHistoryDefaultRounds(t *testing.T) {
	require.Equal(t, conversationHistoryDefaultRounds, newConvHistory(t, nil).cfg.MaxRounds)
	require.Equal(t, 5, newConvHistory(t, map[string]any{"max_rounds": float64(5)}).cfg.MaxRounds)
}

func TestConversationHistoryRawToTextBounded(t *testing.T) {
	s := newConvHistory(t, map[string]any{"max_rounds": float64(2)})

	for _, text := range []string{"one", "two", "three"} {
		_, err := s.RawToText(context.Background(), text)
		require.NoError(t, err)
	}
	require.Len(t, s.messages, 2, "history is bounded to max_rounds")
	require.Equal(t, "two", s.messages[0].Message)
	require.Equal(t, "three", s.messages[1].Message)

	empty, err := s.RawToText(context.Background(), "")
	require.NoError(t, err)
	require.Nil(t, empty, "empty text is ignored")
}

func TestConversationHistoryFormattedBuffer(t *testing.T) {
	s := newConvHistory(t, nil)
	require.Equal(t, "", s.FormattedLatestBuffer(), "empty history yields empty string")

	_, _ = s.RawToText(context.Background(), "hi")
	_, _ = s.RawToText(context.Background(), "bye")
	out := s.FormattedLatestBuffer()
	require.Contains(t, out, "Conversation History")
	require.Contains(t, out, "hi")
	require.Contains(t, out, "bye")
}

func TestConversationHistoryPoll(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.RemoveInput(conversationHistoryVoiceKey)
	t.Cleanup(func() { io.RemoveInput(conversationHistoryVoiceKey); io.ResetTickCounter() })

	s := newConvHistory(t, nil)

	got, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, got)

	io.IncrementTick()
	io.AddInput(conversationHistoryVoiceKey, "  hello  ", time.Now())

	got, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.Equal(t, "hello", got, "voice text is trimmed and returned once per tick")

	got, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, got, "a tick is only emitted once")
}

func TestConversationHistoryStopClears(t *testing.T) {
	s := newConvHistory(t, nil)
	_, _ = s.RawToText(context.Background(), "hi")
	s.Stop()
	require.Empty(t, s.messages)

	got, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, got, "a stopped sensor polls nothing")
}
