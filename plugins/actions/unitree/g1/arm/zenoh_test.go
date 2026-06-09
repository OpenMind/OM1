package arm

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestArmActionEnumValues(t *testing.T) {
	values := ArmAction("").EnumValues()
	require.Contains(t, values, "shake_hand")
	require.Contains(t, values, "stand_still")
	require.NotEmpty(t, values)
}

func TestSerializeUnitreeRequestHeaderAndAPIID(t *testing.T) {
	buf := serializeUnitreeRequest(1042, "")

	require.Equal(t, []byte{0x00, 0x01, 0x00, 0x00}, buf[:4], "CDR encapsulation header")

	require.Equal(t, int64(0), int64(binary.LittleEndian.Uint64(buf[4:12])), "identity.id is zero")
	require.Equal(t, int64(1042), int64(binary.LittleEndian.Uint64(buf[12:20])), "api_id is encoded")
}

func TestSerializeUnitreeRequestParameter(t *testing.T) {
	buf := serializeUnitreeRequest(1, `{"x":1}`)
	require.Contains(t, string(buf), `{"x":1}`, "parameter JSON is embedded")

	require.Zero(t, (len(buf)-4)%4, "payload stays aligned to a 4-byte boundary")
}
