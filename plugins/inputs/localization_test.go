package inputs

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	localizationprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func newTestLocalizationSensor() *LocalizationSensor {
	return &LocalizationSensor{
		log:      zap.NewNop(),
		provider: localizationprovider.NewLocalizationProvider("", 0),
	}
}

func TestLocalizationPoll(t *testing.T) {
	s := newTestLocalizationSensor()

	raw, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Equal(t, notLocalizedText, raw)
}

func TestLocalizationRawToText(t *testing.T) {
	s := newTestLocalizationSensor()

	msg, err := s.RawToText(context.Background(), localizedText)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, localizedText, msg.Message)
	require.NotZero(t, msg.Timestamp)
}

func TestLocalizationRawToTextNonString(t *testing.T) {
	s := newTestLocalizationSensor()

	msg, err := s.RawToText(context.Background(), 42)
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestLocalizationRawToTextEmptyString(t *testing.T) {
	s := newTestLocalizationSensor()

	msg, err := s.RawToText(context.Background(), "")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestLocalizationFormattedLatestBuffer(t *testing.T) {
	s := newTestLocalizationSensor()

	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), notLocalizedText)
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, localizationDescriptor)
	require.Contains(t, out, notLocalizedText)

	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestLocalizationFormattedLatestBufferReturnsNewest(t *testing.T) {
	s := newTestLocalizationSensor()

	_, err := s.RawToText(context.Background(), notLocalizedText)
	require.NoError(t, err)
	_, err = s.RawToText(context.Background(), localizedText)
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, localizedText)
	require.NotContains(t, out, notLocalizedText)
}

func TestLocalizationBoundedHistory(t *testing.T) {
	s := newTestLocalizationSensor()

	for i := 0; i < localizationMaxMessages+5; i++ {
		_, err := s.RawToText(context.Background(), localizedText)
		require.NoError(t, err)
	}

	s.mu.Lock()
	n := len(s.messages)
	s.mu.Unlock()
	require.LessOrEqual(t, n, localizationMaxMessages)
}

func TestLocalizationStopIsIdempotent(t *testing.T) {
	s := newTestLocalizationSensor()

	s.Stop()
	require.True(t, s.stopped)

	require.NotPanics(t, s.Stop)
}

func TestNewLocalization(t *testing.T) {
	s, err := NewLocalization(map[string]any{
		"topic":             "om/localization_pose",
		"quality_tolerance": 0.5,
	})
	require.NoError(t, err)
	require.NotNil(t, s)

	s.Stop()
}
