package greeting_conversation

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSerializePersonGreetingStatusHeader(t *testing.T) {
	buf := serializePersonGreetingStatus("req-1", 2, "ok")

	// CDR encapsulation header.
	require.Equal(t, []byte{0x00, 0x01, 0x00, 0x00}, buf[:4])

	// stamp.sec / stamp.nanosec occupy the next 8 bytes; just assert they're present.
	require.Greater(t, len(buf), 12)

	// frame_id is the request_id as a padded CDR string at data offset 0 (buf[12]).
	frameLen := binary.LittleEndian.Uint32(buf[12:16])
	require.Equal(t, uint32(len("req-1")+1), frameLen, "frame_id length includes the null terminator")
}

func TestSerializePersonGreetingStatusContainsFields(t *testing.T) {
	buf := serializePersonGreetingStatus("abc", 1, "hello")
	require.Contains(t, string(buf), "abc", "request id is encoded")
	require.Contains(t, string(buf), "hello", "message is encoded")
}

func TestSerializePersonGreetingStatusDeterministicLength(t *testing.T) {
	// Same inputs (modulo the embedded timestamp, which doesn't change length)
	// must yield the same buffer length.
	a := serializePersonGreetingStatus("id", 0, "msg")
	b := serializePersonGreetingStatus("id", 0, "msg")
	require.Equal(t, len(a), len(b))
}
