package go2

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func buildLowStatePayload(soc, ntc1, ntc2 byte, powerV, powerA float32) []byte {
	buf := make([]byte, 1180)
	buf[0], buf[1] = 0x00, 0x01

	buf[lowStateSocOffset] = soc
	buf[lowStateNTC1Offset] = ntc1
	buf[lowStateNTC2Offset] = ntc2
	binary.LittleEndian.PutUint32(buf[lowStatePowerVOffset:], math.Float32bits(powerV))
	binary.LittleEndian.PutUint32(buf[lowStatePowerAOffset:], math.Float32bits(powerA))
	return buf
}

func TestOnSampleDecodesBattery(t *testing.T) {
	p := &BatteryZenohProvider{log: zap.NewNop()}

	p.onSample(buildLowStatePayload(88, 41, 43, 33.25, -2.5))

	got := p.State()
	require.InDelta(t, 88.0, got.Percentage, 1e-9)
	require.InDelta(t, 33.25, got.Voltage, 1e-9)
	require.InDelta(t, -2.5, got.Amperes, 1e-9)
	require.Equal(t, 42, got.Temperature)
}

func TestOnSampleShortPayloadIgnored(t *testing.T) {
	p := &BatteryZenohProvider{log: zap.NewNop()}
	p.onSample([]byte{0x00, 0x01, 0x00, 0x00})

	got := p.State()
	require.Zero(t, got.Percentage)
	require.Zero(t, got.Voltage)
}

func TestOnSampleThrottles(t *testing.T) {
	p := &BatteryZenohProvider{log: zap.NewNop()}

	p.onSample(buildLowStatePayload(88, 0, 0, 33.0, 0))
	p.onSample(buildLowStatePayload(50, 0, 0, 30.0, 0))

	got := p.State()
	require.InDelta(t, 88.0, got.Percentage, 1e-9)
	require.InDelta(t, 33.0, got.Voltage, 1e-9)
}
