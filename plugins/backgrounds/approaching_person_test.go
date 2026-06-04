package backgrounds

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPersonGreetingStatusRoundTrip(t *testing.T) {
	cases := []struct {
		name      string
		requestID string
		status    int8
		message   string
	}{
		{"simple", "req-1", 2, "greeting"},
		{"empty strings", "", 0, ""},
		{"max status", "abc123", 127, "done"},
		{"long ids", "request-id-with-some-length", 5, "a longer status message here"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf := serializePersonGreetingStatus(tc.requestID, tc.status, tc.message)
			got, err := deserializePersonGreetingStatus(buf)
			require.NoError(t, err)
			require.Equal(t, tc.status, got, "status survives a serialize/deserialize round trip")
		})
	}
}

func TestDeserializePersonGreetingStatusTooShort(t *testing.T) {
	_, err := deserializePersonGreetingStatus([]byte{0x00, 0x01, 0x00, 0x00})
	require.Error(t, err)
	require.Contains(t, err.Error(), "too short")
}

func TestDeserializePersonGreetingStatusTruncated(t *testing.T) {
	_, err := deserializePersonGreetingStatus(make([]byte, 16))
	require.Error(t, err, "a buffer that ends mid-field is rejected")
}
