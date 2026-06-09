package go2

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	zenohsession "github.com/openmind/om1/internal/zenoh"
)

func buildPoseStampedPayload(sec int32, nanosec uint32, frameID string, pos [3]float64, ori [4]float64) []byte {
	buf := make([]byte, 0, 96)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, sec)
	buf = zenohsession.AppendUint32LE(buf, nanosec)
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

	return buf
}

func TestDeserializePoseStamped(t *testing.T) {
	payload := buildPoseStampedPayload(
		123, 456_000_000, "odom",
		[3]float64{1.5, -2.25, 0.30},
		[4]float64{0, 0, 0, 1},
	)

	ps, err := deserializePoseStamped(payload)
	require.NoError(t, err)
	require.Equal(t, int32(123), ps.stampSec)
	require.Equal(t, uint32(456_000_000), ps.stampNanosec)
	require.InDelta(t, 1.5, ps.posX, 1e-9)
	require.InDelta(t, -2.25, ps.posY, 1e-9)
	require.InDelta(t, 0.30, ps.posZ, 1e-9)
	require.InDelta(t, 1.0, ps.oriW, 1e-9)

	_, err = deserializePoseStamped([]byte{0x00, 0x01})
	require.Error(t, err)
}

func TestEulerFromQuaternion(t *testing.T) {
	_, _, yaw := eulerFromQuaternion(0, 0, 0, 1)
	require.InDelta(t, 0.0, yaw, 1e-9)

	s := math.Sin(math.Pi / 4)
	c := math.Cos(math.Pi / 4)
	_, _, yaw = eulerFromQuaternion(0, 0, s, c)
	require.InDelta(t, math.Pi/2, yaw, 1e-9)
}

func TestProcessOdom(t *testing.T) {
	p := &OdomZenohProvider{log: zap.NewNop()}

	p.processOdom(poseStamped{
		stampSec: 10, stampNanosec: 500_000_000,
		posX: 2.0, posY: 3.0, posZ: 0.30,
		oriW: 1.0,
	})

	got := p.Position()
	require.InDelta(t, 2.0, got.OdomX, 1e-9)
	require.InDelta(t, 3.0, got.OdomY, 1e-9)
	require.Equal(t, 30, got.BodyHeightCm)
	require.Equal(t, RobotStateStanding, got.BodyAttitude)
	require.InDelta(t, 0.0, got.OdomYawM180P180, 1e-9)
	require.InDelta(t, 10.5, got.OdomRockchipTS, 1e-9)
	require.True(t, got.Moving)

	p.processOdom(poseStamped{posX: 2.0, posY: 3.0, posZ: 0.10, oriW: 1.0})
	got = p.Position()
	require.Equal(t, 10, got.BodyHeightCm)
	require.Equal(t, RobotStateSitting, got.BodyAttitude)

	require.GreaterOrEqual(t, got.OdomYaw0360, 0.0)
	require.Less(t, got.OdomYaw0360, 360.0)
}
