package zenoh

import (
	"encoding/binary"
	"fmt"
	"math"
)

// ReadFloat64LE reads a little-endian float64 from data starting at off.
func ReadFloat64LE(data []byte, off int) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(data[off:]))
}

// ReadFloat32LE reads a little-endian float32 from data starting at off.
func ReadFloat32LE(data []byte, off int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
}

// AppendInt32LE appends a little-endian int32 to buf.
func AppendInt32LE(buf []byte, v int32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], uint32(v))

	return append(buf, b[:]...)
}

// AppendUint32LE appends a little-endian uint32 to buf.
func AppendUint32LE(buf []byte, v uint32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)

	return append(buf, b[:]...)
}

// AppendInt64LE appends a little-endian int64 to buf.
func AppendInt64LE(buf []byte, v int64) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(v))

	return append(buf, b[:]...)
}

// AppendFloat64LE appends a little-endian float64 to buf.
func AppendFloat64LE(buf []byte, v float64) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))

	return append(buf, b[:]...)
}

// ReadStdMsgsFloat64 reads a std_msgs/Float64 from data.
func ReadStdMsgsString(data []byte) (string, error) {
	if len(data) < 8 {
		return "", fmt.Errorf("zenoh: std_msgs/String payload too short (%d bytes)", len(data))
	}

	pos := 4
	strLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4

	if pos+strLen > len(data) {
		return "", fmt.Errorf("zenoh: std_msgs/String truncated (need %d bytes, have %d)", pos+strLen, len(data))
	}

	s := data[pos : pos+strLen]
	if len(s) > 0 && s[len(s)-1] == 0x00 {
		s = s[:len(s)-1]
	}

	return string(s), nil
}

// AppendCDRString appends a CDR-encoded string to buf.
func AppendCDRString(buf []byte, s string) []byte {
	strBytes := append([]byte(s), 0x00)
	buf = AppendUint32LE(buf, uint32(len(strBytes)))
	buf = append(buf, strBytes...)

	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	return buf
}
