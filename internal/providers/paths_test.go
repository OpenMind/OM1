package providers

import (
	"testing"

	"github.com/stretchr/testify/require"

	zenohsession "github.com/openmind/om1/internal/zenoh"
)

func buildPathsPayload(paths []uint32) []byte {
	buf := make([]byte, 0, 64)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, 0)
	buf = zenohsession.AppendUint32LE(buf, 0)
	buf = zenohsession.AppendCDRString(buf, "")

	buf = zenohsession.AppendUint32LE(buf, uint32(len(paths)))
	for _, p := range paths {
		buf = zenohsession.AppendUint32LE(buf, p)
	}

	buf = zenohsession.AppendUint32LE(buf, 0)
	buf = zenohsession.AppendUint32LE(buf, 0)
	return buf
}

func TestDeserializePaths(t *testing.T) {
	want := []uint32{0, 3, 4, 5, 9}
	got, err := deserializePaths(buildPathsPayload(want))
	require.NoError(t, err)
	require.Equal(t, want, got)

	empty, err := deserializePaths(buildPathsPayload(nil))
	require.NoError(t, err)
	require.Empty(t, empty)

	_, err = deserializePaths([]byte{0x00, 0x01})
	require.Error(t, err)
}

func TestGenerateMovementString(t *testing.T) {
	classify := func(paths []uint32) string {
		var tl, tr, adv []uint32
		retreat := false
		for _, p := range paths {
			switch {
			case p < 3:
				tl = append(tl, p)
			case p <= 5:
				adv = append(adv, p)
			case p < 9:
				tr = append(tr, p)
			case p == 9:
				retreat = true
			}
		}
		return generateMovementString(paths, tl, adv, tr, retreat)
	}

	require.Equal(t,
		"The safe movement directions are: {'turn left', 'move forwards', 'turn right', 'move back', 'stand still'}. ",
		classify([]uint32{0, 4, 7, 9}),
	)
	require.Equal(t,
		"The safe movement directions are: {'move forwards', 'stand still'}. ",
		classify([]uint32{3, 4, 5}),
	)
	require.Equal(t,
		"The safe movement directions are: {'stand still'}. ",
		classify([]uint32{}),
	)
	require.Equal(t,
		"You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE.",
		generateMovementString(nil, nil, nil, nil, false),
	)
}
