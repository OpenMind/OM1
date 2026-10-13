package asr

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
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

func newHoldCore(hold time.Duration) *asrSensorCore {
	return &asrSensorCore{name: "test", log: zap.NewNop(), transcriptCh: make(chan string, 32), turnHold: hold}
}

func TestHoldTurnJoinsTranscriptsWithinWindow(t *testing.T) {
	a := newHoldCore(120 * time.Millisecond)
	go func() {
		time.Sleep(40 * time.Millisecond)
		a.transcriptCh <- "world"
		time.Sleep(40 * time.Millisecond)
		a.transcriptCh <- "again"
	}()
	start := time.Now()
	got := a.holdTurn(context.Background(), "hello")
	require.Equal(t, "hello world again", got)
	require.GreaterOrEqual(t, time.Since(start), 180*time.Millisecond, "the window restarts after each part")
}

func TestHoldTurnDeliversAloneAfterWindow(t *testing.T) {
	a := newHoldCore(60 * time.Millisecond)
	start := time.Now()
	got := a.holdTurn(context.Background(), "hello")
	require.Equal(t, "hello", got)
	el := time.Since(start)
	require.GreaterOrEqual(t, el, 60*time.Millisecond)
	require.Less(t, el, 400*time.Millisecond)
}

func TestHoldTurnReturnsOnCancel(t *testing.T) {
	a := newHoldCore(5 * time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(30 * time.Millisecond); cancel() }()
	start := time.Now()
	require.Equal(t, "hello", a.holdTurn(ctx, "hello"))
	require.Less(t, time.Since(start), time.Second)
}

func TestPollLoopHoldsTurnBeforeForwarding(t *testing.T) {
	a := newHoldCore(80 * time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := make(chan any, 4)
	go a.pollLoop(ctx, out)
	a.transcriptCh <- "one"
	time.Sleep(30 * time.Millisecond)
	a.transcriptCh <- "two"
	select {
	case raw := <-out:
		require.Equal(t, "one two", raw)
	case <-time.After(2 * time.Second):
		t.Fatal("no output")
	}
	select {
	case raw := <-out:
		t.Fatalf("unexpected second output %v", raw)
	case <-time.After(150 * time.Millisecond):
	}
}

func TestPollLoopNoHoldForwardsImmediately(t *testing.T) {
	a := newHoldCore(0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := make(chan any, 4)
	go a.pollLoop(ctx, out)
	a.transcriptCh <- "one"
	a.transcriptCh <- "two"
	first := <-out
	second := <-out
	require.Equal(t, "one", first)
	require.Equal(t, "two", second)
}

func TestVoiceActiveNilTracker(t *testing.T) {
	var tr *vadLatencyTracker
	require.False(t, tr.VoiceActive())
}
