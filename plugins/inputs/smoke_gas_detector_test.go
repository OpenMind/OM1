package inputs

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func newTestSmokeGasSensor(scenario string) *SmokeGasDetectorSensor {
	return &SmokeGasDetectorSensor{
		log: zap.NewNop(),
		cfg: SmokeGasDetectorConfig{
			CooldownSec:           5.0,
			SmokeWarningThreshold: smokeWarningThresholdDefault,
			SmokeDangerThreshold:  smokeDangerThresholdDefault,
			GasWarningThreshold:   gasWarningThresholdDefault,
			GasDangerThreshold:    gasDangerThresholdDefault,
			MockScenario:          scenario,
		},
		connector: newMockSmokeConnector(scenario),
	}
}

func TestSmokeGasClassify(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	require.Equal(t, "normal", s.classify(&SmokeGasReading{SmokePPM: 100, GasPPM: 100}))
	require.Equal(t, "warning", s.classify(&SmokeGasReading{SmokePPM: 350, GasPPM: 100}))
	require.Equal(t, "danger", s.classify(&SmokeGasReading{SmokePPM: 700, GasPPM: 100}))
	require.Equal(t, "danger", s.classify(&SmokeGasReading{SmokePPM: 100, GasPPM: 700}))
}

func TestSmokeGasReadingToTextNormal(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	text := s.readingToText(&SmokeGasReading{SmokePPM: 50, GasPPM: 40})
	require.Contains(t, text, "Air quality normal")
}

func TestSmokeGasReadingToTextDangerCooldown(t *testing.T) {
	s := newTestSmokeGasSensor("danger")

	first := s.readingToText(&SmokeGasReading{SmokePPM: 750, GasPPM: 700})
	require.Contains(t, first, "SMOKE ALERT")

	second := s.readingToText(&SmokeGasReading{SmokePPM: 750, GasPPM: 700})
	require.Empty(t, second)
}

func TestSmokeGasPoll(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	raw, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Contains(t, raw, "Air quality normal")
}

func TestSmokeGasRawToTextNonString(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	msg, err := s.RawToText(context.Background(), 42)
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestSmokeGasRawToTextEmptyString(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	msg, err := s.RawToText(context.Background(), "")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestSmokeGasFormattedLatestBuffer(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), "Smoke/gas detector: Air quality normal. Smoke: 50 ppm, Gas: 40 ppm.")
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, smokeGasDescriptor)
	require.Contains(t, out, "Air quality normal")

	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestSmokeGasStopIsIdempotent(t *testing.T) {
	s := newTestSmokeGasSensor("normal")

	s.Stop()
	require.True(t, s.stopped)
	require.NotPanics(t, s.Stop)
}

func TestNewSmokeGasDetector(t *testing.T) {
	s, err := NewSmokeGasDetector(map[string]any{
		"mock_scenario": "warning",
		"cooldown":      2.0,
	})
	require.NoError(t, err)
	require.NotNil(t, s)

	s.Stop()
}
