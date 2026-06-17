package backgrounds

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers/tts"
)

func TestTTSRequestRoundTrip(t *testing.T) {
	cases := []struct {
		name      string
		frameID   string
		requestID string
		code      byte
	}{
		{"disable", "om_api", "req-1", ttsCodeDisable},
		{"enable", "frame-2", "req-42", ttsCodeEnable},
		{"status", "f", "r", ttsCodeStatus},
		{"empty ids", "", "", ttsCodeEnable},
		{"long ids", "a-longer-frame-id", "request-id-with-some-length", ttsCodeDisable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf := serializeTTSResponse(tc.frameID, tc.requestID, tc.code, "status text")
			req, err := deserializeTTSRequest(buf)
			require.NoError(t, err)
			require.Equal(t, tc.frameID, req.frameID, "frame_id survives the round trip")
			require.Equal(t, tc.requestID, req.requestID, "request_id survives the round trip")
			require.Equal(t, tc.code, req.code, "code survives the round trip")
		})
	}
}

func TestDeserializeTTSRequestTooShort(t *testing.T) {
	_, err := deserializeTTSRequest([]byte{0x00, 0x01, 0x00, 0x00})
	require.Error(t, err)
	require.Contains(t, err.Error(), "too short")
}

func TestDeserializeTTSRequestTruncated(t *testing.T) {
	_, err := deserializeTTSRequest(make([]byte, 16))
	require.Error(t, err, "a buffer that ends mid-field is rejected")
}

func TestOnTTSRequestTogglesSuppression(t *testing.T) {
	t.Cleanup(func() { tts.Suppressed.Store(false) })

	ctrl := &TTSControl{log: zap.NewNop()}

	cases := []struct {
		name  string
		start bool
		code  byte
		want  bool
	}{
		{"disable mutes", false, ttsCodeDisable, true},
		{"enable unmutes", true, ttsCodeEnable, false},
		{"status leaves muted state", true, ttsCodeStatus, true},
		{"status leaves unmuted state", false, ttsCodeStatus, false},
		{"unknown code is ignored", true, 99, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tts.Suppressed.Store(tc.start)
			ctrl.onTTSRequest(serializeTTSResponse("om_api", "req", tc.code, ""))
			require.Equal(t, tc.want, tts.Suppressed.Load())
		})
	}
}

func TestOnTTSRequestInvalidPayloadIsIgnored(t *testing.T) {
	t.Cleanup(func() { tts.Suppressed.Store(false) })
	tts.Suppressed.Store(true)

	ctrl := &TTSControl{log: zap.NewNop()}
	ctrl.onTTSRequest([]byte{0x00, 0x01})

	require.True(t, tts.Suppressed.Load(), "suppression unchanged on decode failure")
}
