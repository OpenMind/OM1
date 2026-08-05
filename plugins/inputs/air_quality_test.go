package inputs

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/plugins/inputs/air_quality/connector"
)

// mockAQConnector is a test double implementing connector.AirQualityConnector.
type mockAQConnector struct {
	connectOK   bool
	connectErr  error
	readData    *connector.AirQualityData
	readErr     error
	disconnectErr error
	connectCalls int
	readCalls    int
	disconnectCalls int
}

func (m *mockAQConnector) Connect(ctx context.Context) (bool, error) {
	m.connectCalls++
	return m.connectOK, m.connectErr
}

func (m *mockAQConnector) Read(ctx context.Context) (*connector.AirQualityData, error) {
	m.readCalls++
	return m.readData, m.readErr
}

func (m *mockAQConnector) Disconnect(ctx context.Context) error {
	m.disconnectCalls++
	return m.disconnectErr
}

func (m *mockAQConnector) Name() string {
	return "mock"
}

func intPtr(i int) *int             { return &i }
func f64Ptr(f float64) *float64     { return &f }

func newTestAirQualitySensor(conn connector.AirQualityConnector) *AirQualitySensor {
	return &AirQualitySensor{
		log:  zap.NewNop(),
		conn: conn,
		cfg: AirQualityConfig{
			Connector:           "mock",
			PollIntervalSec:     airQualityDefaultPollIntervalSec,
			AQIWarningThreshold: airQualityDefaultAQIWarningThreshold,
			AQIDangerThreshold:  airQualityDefaultAQIDangerThreshold,
		},
	}
}

func TestAirQualityPollOnce_Success(t *testing.T) {
	aqi := 42
	pm25 := 12.5
	mock := &mockAQConnector{
		connectOK: true,
		readData: &connector.AirQualityData{
			AQI:      &aqi,
			PM25:     &pm25,
			Location: "Semarang, Indonesia",
			Source:   "mock",
		},
	}
	s := newTestAirQualitySensor(mock)

	data, err := s.pollOnce(context.Background())
	require.NoError(t, err)
	require.NotNil(t, data)
	require.Equal(t, 42, *data.AQI)
	require.Equal(t, 1, mock.connectCalls)
	require.Equal(t, 1, mock.readCalls)
	require.Equal(t, 1, mock.disconnectCalls)
}

func TestAirQualityPollOnce_ConnectFailed(t *testing.T) {
	mock := &mockAQConnector{connectOK: false}
	s := newTestAirQualitySensor(mock)

	data, err := s.pollOnce(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
	require.Equal(t, 1, mock.connectCalls)
	require.Equal(t, 0, mock.readCalls) // must not read if connect failed
}

func TestAirQualityPollOnce_TooSoon(t *testing.T) {
	mock := &mockAQConnector{connectOK: true}
	s := newTestAirQualitySensor(mock)
	s.lastPollTime = time.Now()

	data, err := s.pollOnce(context.Background())
	require.NoError(t, err)
	require.Nil(t, data)
	require.Equal(t, 0, mock.connectCalls) // must not even connect
}

func TestAirQualityRawToText_NilData(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	msg, err := s.RawToText(context.Background(), nil)
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestAirQualityRawToText_WrongType(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	msg, err := s.RawToText(context.Background(), "not air quality data")
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestAirQualityRawToText_FullData(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	data := &connector.AirQualityData{
		AQI:         intPtr(99),
		PM25:        f64Ptr(99),
		Temperature: f64Ptr(32.9),
		Humidity:    f64Ptr(53.1),
		Location:    "Semarang, Indonesia",
		Source:      "aqicn",
	}

	msg, err := s.RawToText(context.Background(), data)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Contains(t, msg.Message, "Air Quality in Semarang, Indonesia: MODERATE (AQI: 99)")
	require.Contains(t, msg.Message, "PM2.5: 99")
	require.Contains(t, msg.Message, "Temperature: 32.9°C")
	require.Contains(t, msg.Message, "Humidity: 53.1%")
	require.NotZero(t, msg.Timestamp)
}

func TestAirQualityRawToText_WarningThreshold(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	data := &connector.AirQualityData{AQI: intPtr(120), Location: "Jakarta"}

	msg, err := s.RawToText(context.Background(), data)
	require.NoError(t, err)
	require.Contains(t, msg.Message, "WARNING: Air quality is")
	require.NotContains(t, msg.Message, "DANGER")
}

func TestAirQualityRawToText_DangerThreshold(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	data := &connector.AirQualityData{AQI: intPtr(200), Location: "Jakarta"}

	msg, err := s.RawToText(context.Background(), data)
	require.NoError(t, err)
	require.Contains(t, msg.Message, "DANGER: Air quality is")
}

func TestAirQualityRawToText_NoAQI(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	data := &connector.AirQualityData{
		Temperature: f64Ptr(25.0),
		Location:    "Robot",
	}

	msg, err := s.RawToText(context.Background(), data)
	require.NoError(t, err)
	require.Contains(t, msg.Message, "Air Quality in Robot")
	require.NotContains(t, msg.Message, "AQI:")
}

func TestAirQualityFormattedLatestBuffer(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	require.Equal(t, "", s.FormattedLatestBuffer())

	data := &connector.AirQualityData{AQI: intPtr(50), Location: "Robot"}
	_, err := s.RawToText(context.Background(), data)
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, "INPUT: "+airQualityDescriptor)
	require.Contains(t, out, "// START")
	require.Contains(t, out, "// END")
	require.Contains(t, out, "Robot")

	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestAirQualityFormattedLatestBufferReturnsNewest(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})

	_, err := s.RawToText(context.Background(), &connector.AirQualityData{AQI: intPtr(10), Location: "Old"})
	require.NoError(t, err)
	_, err = s.RawToText(context.Background(), &connector.AirQualityData{AQI: intPtr(20), Location: "New"})
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, "New")
	require.NotContains(t, out, "Old")
}

func TestAirQualityBoundedHistory(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	for i := 0; i < airQualityMaxMessages+5; i++ {
		_, err := s.RawToText(context.Background(), &connector.AirQualityData{AQI: intPtr(10), Location: "Robot"})
		require.NoError(t, err)
	}
	s.mu.Lock()
	n := len(s.messages)
	s.mu.Unlock()
	require.LessOrEqual(t, n, airQualityMaxMessages)
}

func TestAirQualityStopIsIdempotent(t *testing.T) {
	s := newTestAirQualitySensor(&mockAQConnector{})
	s.Stop()
	require.True(t, s.stopped)
	require.NotPanics(t, s.Stop)
}

func TestNewAirQuality_DefaultsToAqicn(t *testing.T) {
	s, err := NewAirQuality(map[string]any{})
	require.NoError(t, err)
	require.NotNil(t, s)
	s.Stop()
}

func TestNewAirQuality_UnknownConnector(t *testing.T) {
	s, err := NewAirQuality(map[string]any{"connector": "nonexistent"})
	require.Error(t, err)
	require.Nil(t, s)
}

func TestNewAirQuality_Pms5003(t *testing.T) {
	s, err := NewAirQuality(map[string]any{
		"connector": "pms5003",
		"connector_config": map[string]any{
			"port":     "/dev/ttyUSB0",
			"location": "Outdoor",
		},
	})
	require.NoError(t, err)
	require.NotNil(t, s)
	s.Stop()
}

func TestNewAirQuality_Bme680(t *testing.T) {
	s, err := NewAirQuality(map[string]any{
		"connector": "bme680",
		"connector_config": map[string]any{
			"i2c_address": 0x76,
			"location":    "Indoor",
		},
	})
	require.NoError(t, err)
	require.NotNil(t, s)
	s.Stop()
}
