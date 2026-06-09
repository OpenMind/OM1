package util

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSleepCompletes(t *testing.T) {
	start := time.Now()
	ok := Sleep(context.Background(), 10*time.Millisecond)
	require.True(t, ok, "Sleep returns true when the timer elapses")
	require.GreaterOrEqual(t, time.Since(start), 10*time.Millisecond)
}

func TestSleepCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	ok := Sleep(ctx, time.Hour)
	require.False(t, ok, "Sleep returns false when the context is cancelled before the timer")
}

func TestSleepCancelledMidWait(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	start := time.Now()
	ok := Sleep(ctx, time.Hour)
	require.False(t, ok)
	require.Less(t, time.Since(start), time.Second, "returns promptly when ctx is cancelled")
}
