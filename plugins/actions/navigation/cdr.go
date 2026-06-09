package navigation

import (
	"encoding/binary"
	"fmt"
	"time"

	"github.com/openmind/om1/internal/geometry"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

// serializePoseStamped encodes a geometry_msgs/PoseStamped message in CDR
// little-endian format, ready to publish on the Nav2 /goal_pose topic.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  header.stamp.sec      int32  LE  (data offset 0)
//	[8]  header.stamp.nanosec  uint32 LE  (data offset 4)
//	[12] header.frame_id       CDR string (data offset 8) + padding to 4 bytes
//	[..] padding to 8-byte data boundary (float64 alignment)
//	[..] pose.position.x/y/z          float64 LE x3
//	[..] pose.orientation.x/y/z/w     float64 LE x4
func serializePoseStamped(frameID string, pose geometry.Pose) []byte {
	now := time.Now()
	return serializePoseStampedAt(int32(now.Unix()), uint32(now.Nanosecond()), frameID, pose)
}

// serializePoseStampedAt is serializePoseStamped with an explicit stamp.
func serializePoseStampedAt(sec int32, nanosec uint32, frameID string, pose geometry.Pose) []byte {
	buf := make([]byte, 0, 4+8+16+7*8)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, sec)
	buf = zenohsession.AppendUint32LE(buf, nanosec)
	buf = zenohsession.AppendCDRString(buf, frameID) // header.frame_id, padded to 4

	if dataLen := len(buf) - 4; (8-dataLen%8)%8 > 0 {
		buf = append(buf, make([]byte, (8-dataLen%8)%8)...)
	}

	buf = zenohsession.AppendFloat64LE(buf, pose.Position.X)
	buf = zenohsession.AppendFloat64LE(buf, pose.Position.Y)
	buf = zenohsession.AppendFloat64LE(buf, pose.Position.Z)
	buf = zenohsession.AppendFloat64LE(buf, pose.Orientation.X)
	buf = zenohsession.AppendFloat64LE(buf, pose.Orientation.Y)
	buf = zenohsession.AppendFloat64LE(buf, pose.Orientation.Z)
	buf = zenohsession.AppendFloat64LE(buf, pose.Orientation.W)

	return buf
}

// serializeAIStatusRequest encodes an AIStatusRequest message in CDR
// little-endian format. It is the request counterpart of the response encoded
// by the autonomy connector: header + request_id String + code int8.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte data boundary
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
func serializeAIStatusRequest(frameID, requestID string, code byte) []byte {
	now := time.Now()
	return serializeAIStatusRequestAt(int32(now.Unix()), uint32(now.Nanosecond()), frameID, requestID, code)
}

// serializeAIStatusRequestAt is serializeAIStatusRequest with an explicit stamp.
func serializeAIStatusRequestAt(sec int32, nanosec uint32, frameID, requestID string, code byte) []byte {
	buf := make([]byte, 0, 64)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, sec)
	buf = zenohsession.AppendUint32LE(buf, nanosec)
	buf = zenohsession.AppendCDRString(buf, frameID) // header.frame_id, padded

	// request_id String: length uint32 + bytes (no padding — next field is int8).
	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	// code: int8
	buf = append(buf, code)

	return buf
}

const (
	statusUnknown   int32 = 0
	statusAccepted  int32 = 1
	statusExecuting int32 = 2
	statusCanceling int32 = 3
	statusSucceeded int32 = 4
	statusCanceled  int32 = 5
	statusAborted   int32 = 6
)

// goalStatusSize is the size of each GoalStatus element in the Nav2Status status_list, in bytes.
const goalStatusSize = 16 + 4 + 4 + 4

// parseNav2StatusLatest extracts the latest goal status code from a CDR-encoded Nav2Status message.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0] CDR encapsulation header: 4 bytes
//	[4] status_list length        uint32 LE
//	[8] status_list[0..n] each goalStatusSize bytes:
//	      uuid[16] | stamp.sec int32 | stamp.nanosec int32 | status int32
func parseNav2StatusLatest(data []byte) (int32, bool) {
	if len(data) < 8 {
		return statusUnknown, false
	}

	count := int(binary.LittleEndian.Uint32(data[4:]))
	if count <= 0 {
		return statusUnknown, false
	}

	// Offset of the last element's status field.
	statusOff := 8 + (count-1)*goalStatusSize + 16 + 4 + 4
	if statusOff+4 > len(data) {
		return statusUnknown, false
	}

	return int32(binary.LittleEndian.Uint32(data[statusOff:])), true
}

// statusName maps a Nav2 goal status code to its human-readable name.
func statusName(code int32) string {
	switch code {
	case statusAccepted:
		return "ACCEPTED"
	case statusExecuting:
		return "EXECUTING"
	case statusCanceling:
		return "CANCELING"
	case statusSucceeded:
		return "SUCCEEDED"
	case statusCanceled:
		return "CANCELED"
	case statusAborted:
		return "ABORTED"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", code)
	}
}
