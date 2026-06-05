package zenoh

import (
	"encoding/binary"
	"math"
)

// ReadFloat64LE reads a little-endian float64 from data starting at off.
func ReadFloat64LE(data []byte, off int) float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(data[off:]))
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
