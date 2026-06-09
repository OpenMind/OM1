package navigation

import (
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/geometry"
)

func TestSerializePoseStampedMatchesPycdr2(t *testing.T) {
	pose := geometry.Pose{
		Position:    geometry.Point{X: 1.0, Y: 2.0, Z: 3.0},
		Orientation: geometry.Quaternion{X: 0, Y: 0, Z: 0, W: 1.0},
	}

	got := serializePoseStampedAt(0x11223344, 0x55667788, "map", pose)

	const golden = "000100004433221188776655040000006d617000" +
		"000000000000f03f00000000000000400000000000000840" +
		"000000000000000000000000000000000000000000000000000000000000f03f"
	assert.Equal(t, golden, hex.EncodeToString(got))
}

func TestSerializeAIStatusRequestMatchesPycdr2(t *testing.T) {
	got := serializeAIStatusRequestAt(0x11223344, 0x55667788, "map", "ab", 0)

	const golden = "000100004433221188776655040000006d6170000300000061620000"
	assert.Equal(t, golden, hex.EncodeToString(got))
}

func TestParseNav2StatusLatestReturnsLastStatus(t *testing.T) {
	// Nav2Status with two GoalStatus entries: status 4 (SUCCEEDED) then 6 (ABORTED).
	raw, err := hex.DecodeString("00010000" + "02000000" +
		"00000000000000000000000000000000" + "01000000" + "02000000" + "04000000" +
		"00000000000000000000000000000000" + "01000000" + "02000000" + "06000000")
	require.NoError(t, err)

	code, ok := parseNav2StatusLatest(raw)
	require.True(t, ok)
	assert.Equal(t, statusAborted, code)
}

func TestParseNav2StatusLatestEmptyList(t *testing.T) {
	raw, err := hex.DecodeString("00010000" + "00000000")
	require.NoError(t, err)

	_, ok := parseNav2StatusLatest(raw)
	assert.False(t, ok)
}

func TestParseNav2StatusLatestTruncated(t *testing.T) {
	// Claims one element but the buffer is too short to hold it.
	raw, err := hex.DecodeString("00010000" + "01000000" + "0000")
	require.NoError(t, err)

	_, ok := parseNav2StatusLatest(raw)
	assert.False(t, ok)
}

func TestStatusName(t *testing.T) {
	assert.Equal(t, "SUCCEEDED", statusName(statusSucceeded))
	assert.Equal(t, "ABORTED", statusName(statusAborted))
	assert.Equal(t, "UNKNOWN(42)", statusName(42))
}
