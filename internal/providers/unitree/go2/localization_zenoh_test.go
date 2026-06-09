package go2

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	zenohsession "github.com/openmind/om1/internal/zenoh"
)

func buildLocalizationPayload(frameID string, pos [3]float64, ori [4]float64, matchScore int32, quality float32, numPoints int32) []byte {
	buf := make([]byte, 0, 128)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, 1)
	buf = zenohsession.AppendInt32LE(buf, 2)
	buf = zenohsession.AppendCDRString(buf, frameID)

	if dataOff := len(buf) - 4; (8-dataOff%8)%8 > 0 {
		buf = append(buf, make([]byte, (8-dataOff%8)%8)...)
	}

	buf = zenohsession.AppendFloat64LE(buf, pos[0])
	buf = zenohsession.AppendFloat64LE(buf, pos[1])
	buf = zenohsession.AppendFloat64LE(buf, pos[2])
	buf = zenohsession.AppendFloat64LE(buf, ori[0])
	buf = zenohsession.AppendFloat64LE(buf, ori[1])
	buf = zenohsession.AppendFloat64LE(buf, ori[2])
	buf = zenohsession.AppendFloat64LE(buf, ori[3])

	buf = zenohsession.AppendInt32LE(buf, matchScore)

	var f32 [4]byte
	binary.LittleEndian.PutUint32(f32[:], math.Float32bits(quality))
	buf = append(buf, f32[:]...)

	buf = zenohsession.AppendInt32LE(buf, numPoints)

	return buf
}

func TestDeserializeLocalization(t *testing.T) {
	payload := buildLocalizationPayload(
		"map",
		[3]float64{1.5, -2.25, 0.30},
		[4]float64{0, 0, 0, 1},
		42, 0.85, 1000,
	)

	m, err := deserializeLocalization(payload)
	require.NoError(t, err)
	require.InDelta(t, 1.5, m.pose.Position.X, 1e-9)
	require.InDelta(t, -2.25, m.pose.Position.Y, 1e-9)
	require.InDelta(t, 0.30, m.pose.Position.Z, 1e-9)
	require.InDelta(t, 1.0, m.pose.Orientation.W, 1e-9)
	require.Equal(t, int32(42), m.matchScore)
	require.InDelta(t, 0.85, float64(m.qualityPercent), 1e-6)
	require.Equal(t, int32(1000), m.numPoints)

	_, err = deserializeLocalization([]byte{0x00, 0x01})
	require.Error(t, err)
}

func TestLocalizationProcess(t *testing.T) {
	p := &LocalizationProvider{log: zap.NewNop(), qualityTolerance: defaultQualityTolerance}

	p.process(localization{
		pose:           Pose{Position: Point{X: 1, Y: 2, Z: 3}},
		qualityPercent: 0.5,
	})
	require.False(t, p.IsLocalized())
	require.NotNil(t, p.Pose())
	require.InDelta(t, 1.0, p.Pose().Position.X, 1e-9)

	p.process(localization{
		pose:           Pose{Position: Point{X: 4, Y: 5, Z: 6}},
		qualityPercent: 0.7,
	})
	require.True(t, p.IsLocalized())
	require.InDelta(t, 4.0, p.Pose().Position.X, 1e-9)
}

func TestLocalizationPoseNilByDefault(t *testing.T) {
	p := &LocalizationProvider{log: zap.NewNop(), qualityTolerance: defaultQualityTolerance}
	require.False(t, p.IsLocalized())
	require.Nil(t, p.Pose())
}
