package logger

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func TestGetReturnsNonNilByDefault(t *testing.T) {
	require.NotNil(t, Get(), "init() installs a production logger")
}

func TestSetAndGet(t *testing.T) {
	original := Get()
	t.Cleanup(func() { Set(original) })

	custom := zap.NewNop()
	Set(custom)
	require.Same(t, custom, Get())
}

func TestBuildLoggerLevels(t *testing.T) {
	cases := []struct {
		level string
		want  zapcore.Level
	}{
		{"debug", zapcore.DebugLevel},
		{"info", zapcore.InfoLevel},
		{"warn", zapcore.WarnLevel},
		{"error", zapcore.ErrorLevel},
	}
	for _, tc := range cases {
		t.Run(tc.level, func(t *testing.T) {
			l := BuildLogger(tc.level)
			require.NotNil(t, l)
			require.True(t, l.Core().Enabled(tc.want))
		})
	}
}

func TestBuildLoggerInvalidLevelDefaultsToInfo(t *testing.T) {
	l := BuildLogger("not-a-level")
	require.NotNil(t, l)
	require.True(t, l.Core().Enabled(zapcore.InfoLevel))
	require.False(t, l.Core().Enabled(zapcore.DebugLevel), "invalid level falls back to info, not debug")
}
