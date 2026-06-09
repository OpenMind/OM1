package inputs

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/geometry"
)

type fakeLocalizationProvider struct {
	localized bool
	pose      *geometry.Pose
	stopped   bool
}

func (f *fakeLocalizationProvider) IsLocalized() bool    { return f.localized }
func (f *fakeLocalizationProvider) Pose() *geometry.Pose { return f.pose }
func (f *fakeLocalizationProvider) Stop()                { f.stopped = true }

func newTestLocalizationSensor(p localizationProvider) *LocalizationSensor {
	return &LocalizationSensor{log: zap.NewNop(), provider: p}
}

func TestLocalizationPoll(t *testing.T) {
	pose := &geometry.Pose{Position: geometry.Point{X: 1, Y: 2, Z: 3}}

	cases := []struct {
		name     string
		provider *fakeLocalizationProvider
		want     string
	}{
		{"localized with pose", &fakeLocalizationProvider{localized: true, pose: pose}, localizedText},
		{"localized but no pose", &fakeLocalizationProvider{localized: true, pose: nil}, notLocalizedText},
		{"not localized with pose", &fakeLocalizationProvider{localized: false, pose: pose}, notLocalizedText},
		{"not localized no pose", &fakeLocalizationProvider{localized: false, pose: nil}, notLocalizedText},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newTestLocalizationSensor(tc.provider)
			raw, err := s.Poll(context.Background())
			require.NoError(t, err)
			require.Equal(t, tc.want, raw)
		})
	}
}

func TestLocalizationRawToText(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	msg, err := s.RawToText(context.Background(), localizedText)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, localizedText, msg.Message)
	require.NotZero(t, msg.Timestamp)
}

func TestLocalizationRawToTextNonString(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	msg, err := s.RawToText(context.Background(), 42)
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestLocalizationRawToTextEmptyString(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	msg, err := s.RawToText(context.Background(), "")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestLocalizationFormattedLatestBuffer(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), notLocalizedText)
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, localizationDescriptor)
	require.Contains(t, out, notLocalizedText)

	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestLocalizationFormattedLatestBufferReturnsNewest(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	_, err := s.RawToText(context.Background(), notLocalizedText)
	require.NoError(t, err)
	_, err = s.RawToText(context.Background(), localizedText)
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, localizedText)
	require.NotContains(t, out, notLocalizedText)
}

func TestLocalizationBoundedHistory(t *testing.T) {
	s := newTestLocalizationSensor(&fakeLocalizationProvider{})

	for i := 0; i < localizationMaxMessages+5; i++ {
		_, err := s.RawToText(context.Background(), localizedText)
		require.NoError(t, err)
	}

	s.mu.Lock()
	n := len(s.messages)
	s.mu.Unlock()
	require.LessOrEqual(t, n, localizationMaxMessages)
}

func TestLocalizationStopIsIdempotentAndStopsProvider(t *testing.T) {
	p := &fakeLocalizationProvider{}
	s := newTestLocalizationSensor(p)

	s.Stop()
	require.True(t, p.stopped)
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
