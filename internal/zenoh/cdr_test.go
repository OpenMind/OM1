package zenoh

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAppendInt32LE(t *testing.T) {
	buf := AppendInt32LE(nil, -2)
	require.Len(t, buf, 4)
	require.Equal(t, int32(-2), int32(binary.LittleEndian.Uint32(buf)))
}

func TestAppendUint32LE(t *testing.T) {
	buf := AppendUint32LE([]byte{0xAA}, 0x01020304)
	require.Equal(t, byte(0xAA), buf[0], "existing bytes are preserved")
	require.Equal(t, uint32(0x01020304), binary.LittleEndian.Uint32(buf[1:]))
}

func TestAppendInt64LE(t *testing.T) {
	buf := AppendInt64LE(nil, -1)
	require.Len(t, buf, 8)
	require.Equal(t, int64(-1), int64(binary.LittleEndian.Uint64(buf)))
}

func TestAppendCDRString(t *testing.T) {
	header := []byte{0x00, 0x01, 0x00, 0x00}
	buf := AppendCDRString(append([]byte(nil), header...), "hi")

	length := binary.LittleEndian.Uint32(buf[4:8])
	require.Equal(t, uint32(3), length)
	require.Equal(t, []byte("hi\x00"), buf[8:11])

	dataLen := len(buf) - 4
	require.Zero(t, dataLen%4, "data payload is aligned to a 4-byte boundary")
	require.Equal(t, byte(0x00), buf[11], "one pad byte follows the 3-byte string")
}

func TestAppendCDRStringAlreadyAligned(t *testing.T) {
	header := []byte{0x00, 0x01, 0x00, 0x00}
	buf := AppendCDRString(append([]byte(nil), header...), "abc")
	require.Equal(t, uint32(4), binary.LittleEndian.Uint32(buf[4:8]))
	require.Len(t, buf, 4+4+4, "header + length + 4 string bytes, no padding")
}
