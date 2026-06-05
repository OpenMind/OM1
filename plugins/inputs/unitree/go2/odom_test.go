package go2

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	odomprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func newTestSensor() *OdomSensor {
	return &OdomSensor{log: zap.NewNop()}
}

func TestRawToTextAttitudes(t *testing.T) {
	cases := []struct {
		name string
		pos  odomprovider.OdomPosition
		want string
	}{
		{"sitting", odomprovider.OdomPosition{BodyAttitude: odomprovider.RobotStateSitting}, odomSittingText},
		{"moving", odomprovider.OdomPosition{Moving: true, BodyAttitude: odomprovider.RobotStateStanding}, odomMovingText},
		{"standing", odomprovider.OdomPosition{BodyAttitude: odomprovider.RobotStateStanding}, odomStandingText},
		{"sitting overrides moving", odomprovider.OdomPosition{Moving: true, BodyAttitude: odomprovider.RobotStateSitting}, odomSittingText},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestSensor()
			msg, err := s.RawToText(context.Background(), tc.pos)
			require.NoError(t, err)
			require.NotNil(t, msg)
			require.Equal(t, tc.want, msg.Message)
		})
	}
}

func TestRawToTextWrongType(t *testing.T) {
	s := newTestSensor()
	msg, err := s.RawToText(context.Background(), "not a position")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestFormattedLatestBuffer(t *testing.T) {
	s := newTestSensor()
	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), odomprovider.OdomPosition{BodyAttitude: odomprovider.RobotStateSitting})
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, odomDescriptor)
	require.Contains(t, out, odomSittingText)

	// Buffer is cleared after formatting.
	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestRawToTextBoundedHistory(t *testing.T) {
	s := newTestSensor()
	for i := 0; i < odomMaxMessages+5; i++ {
		_, err := s.RawToText(context.Background(), odomprovider.OdomPosition{BodyAttitude: odomprovider.RobotStateStanding})
		require.NoError(t, err)
	}
	s.mu.Lock()
	n := len(s.messages)
	s.mu.Unlock()
	require.LessOrEqual(t, n, odomMaxMessages)
}
