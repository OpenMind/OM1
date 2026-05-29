package zenoh

import "encoding/binary"

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

// AppendCDRString writes a CDR string: uint32 length (including null terminator),
// string bytes + null, then pads to a 4-byte boundary measured from the start of
// the data payload (i.e. buf[4:], after the 4-byte CDR encapsulation header).
func AppendCDRString(buf []byte, s string) []byte {
	strBytes := append([]byte(s), 0x00)
	buf = AppendUint32LE(buf, uint32(len(strBytes)))
	buf = append(buf, strBytes...)
	// align to 4-byte data boundary (data starts after the 4-byte CDR header)
	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}
	return buf
}
