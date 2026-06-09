package greeting_conversation

import (
	"time"

	"github.com/openmind/om1/internal/zenoh"
)

// serializePersonGreetingStatus encodes a PersonGreetingStatus in CDR
// little-endian format.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8) + padding to 4-byte
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] status           int8
//	[..] padding to 4-byte data boundary (before message uint32 length)
//	[..] message          length uint32 + bytes+NUL (last field, no padding)
func serializePersonGreetingStatus(requestID string, status byte, message string) []byte {
	now := time.Now()

	buf := make([]byte, 0, 256)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenoh.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenoh.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenoh.AppendCDRString(buf, requestID) // frame_id, padded

	// request_id String: length uint32 + bytes (no padding — next field is int8)
	reqBytes := append([]byte(requestID), 0x00)
	buf = zenoh.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	// status: int8
	buf = append(buf, status)

	// padding to 4-byte data boundary (before message length uint32)
	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	// message String: length uint32 + bytes (last field, no padding)
	msgBytes := append([]byte(message), 0x00)
	buf = zenoh.AppendUint32LE(buf, uint32(len(msgBytes)))
	buf = append(buf, msgBytes...)

	return buf
}
