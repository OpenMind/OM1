package autonomy

import (
	"encoding/binary"
	"fmt"
	"time"

	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// AIStatusRequest.Code values.
const (
	aiCodeDisabled byte = 0
	aiCodeEnabled  byte = 1
	aiCodeStatus   byte = 2
)

// aiStatusRequest is the decoded payload of an AIStatusRequest message.
type aiStatusRequest struct {
	frameID   string
	requestID string
	code      byte
}

// deserializeAIStatusRequest decodes the header frame_id, request_id, and code
// fields from a CDR-encoded AIStatusRequest payload.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 4 bytes
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte data boundary
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
func deserializeAIStatusRequest(data []byte) (aiStatusRequest, error) {
	var req aiStatusRequest

	if len(data) < 16 {
		return req, fmt.Errorf("ai status: payload too short (%d bytes)", len(data))
	}

	pos := 4 // skip CDR header
	pos += 4 // stamp.sec
	pos += 4 // stamp.nanosec

	frameID, next, err := readCDRString(data, pos, true)
	if err != nil {
		return req, fmt.Errorf("ai status: frame_id: %w", err)
	}
	req.frameID = frameID
	pos = next

	requestID, next, err := readCDRString(data, pos, false)
	if err != nil {
		return req, fmt.Errorf("ai status: request_id: %w", err)
	}
	req.requestID = requestID
	pos = next

	if pos >= len(data) {
		return req, fmt.Errorf("ai status: truncated at code")
	}
	req.code = data[pos]

	return req, nil
}

func readCDRString(data []byte, off int, padded bool) (string, int, error) {
	if off+4 > len(data) {
		return "", off, fmt.Errorf("truncated at length")
	}
	strLen := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if off+strLen > len(data) {
		return "", off, fmt.Errorf("truncated at data (len=%d)", strLen)
	}

	b := data[off : off+strLen]
	if strLen > 0 && b[strLen-1] == 0 {
		b = b[:strLen-1]
	}
	off += strLen

	if padded {
		dataOff := off - 4
		off += (4 - dataOff%4) % 4
	}

	return string(b), off, nil
}

// serializeAIStatusResponse encodes an AIStatusResponse message in CDR
// little-endian format.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte data boundary
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
//	[..] padding to 4-byte data boundary (before status uint32 length)
//	[..] status           length uint32 + bytes+NUL (last field, no padding)
func serializeAIStatusResponse(frameID, requestID string, code byte, status string) []byte {
	now := time.Now()

	buf := make([]byte, 0, 128)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenohsession.AppendCDRString(buf, frameID) // header.frame_id, padded

	// request_id String: length uint32 + bytes (no padding — next field is int8)
	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	// code: int8
	buf = append(buf, code)

	// padding to 4-byte data boundary (before status length uint32)
	dataLen := len(buf) - 4 // padding is measured from the start of CDR data, not the buffer
	buf = append(buf, make([]byte, (4-dataLen%4)%4)...)

	// status String: length uint32 + bytes (last field, no padding)
	statusBytes := append([]byte(status), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(statusBytes)))
	buf = append(buf, statusBytes...)

	return buf
}
