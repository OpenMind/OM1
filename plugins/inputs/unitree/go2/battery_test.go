package go2

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	batteryprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func newTestBatterySensor() *BatterySensor {
	return &BatterySensor{log: zap.NewNop()}
}

func TestBatteryRawToTextThresholds(t *testing.T) {
	cases := []struct {
		name    string
		pct     float64
		wantMsg bool
		want    string
	}{
		{"critical", 5, true, batteryCriticalText},
		{"critical boundary", 6.99, true, batteryCriticalText},
		{"warning", 12, true, batteryWarningText},
		{"warning boundary", 14.99, true, batteryWarningText},
		{"healthy", 80, false, ""},
		{"exactly warning threshold", 15, false, ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestBatterySensor()
			msg, err := s.RawToText(context.Background(), batteryprovider.BatteryState{Percentage: tc.pct})
			require.NoError(t, err)
			if tc.wantMsg {
				require.NotNil(t, msg)
				require.Equal(t, tc.want, msg.Message)
			} else {
				require.Nil(t, msg)
			}
		})
	}
}

func TestBatteryRawToTextWrongType(t *testing.T) {
	s := newTestBatterySensor()
	msg, err := s.RawToText(context.Background(), "not a battery state")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestBatteryFormattedLatestBuffer(t *testing.T) {
	s := newTestBatterySensor()
	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), batteryprovider.BatteryState{Percentage: 3})
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, batteryDescriptor)
	require.Contains(t, out, batteryCriticalText)

	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestBatteryRawToTextBoundedHistory(t *testing.T) {
	s := newTestBatterySensor()
	for i := 0; i < batteryMaxMessages+5; i++ {
		_, err := s.RawToText(context.Background(), batteryprovider.BatteryState{Percentage: 3})
		require.NoError(t, err)
	}
	s.mu.Lock()
	n := len(s.messages)
	s.mu.Unlock()
	require.LessOrEqual(t, n, batteryMaxMessages)
}
