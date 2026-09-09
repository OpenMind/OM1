package vlm

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestVideoRTSPDefaultURLValue(t *testing.T) {
	require.Equal(t, "rtsp://localhost:8556/raw", defaultRTSPURL)

	v := NewVideoRTSPStream(VideoRTSPStreamConfig{})
	require.Equal(t, "rtsp://localhost:8556/raw", v.cfg.RTSPURL)
}

func TestNewVideoRTSPStreamPreservesExplicitURL(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{RTSPURL: "rtsp://example.test/live"})
	require.Equal(t, "rtsp://example.test/live", v.cfg.RTSPURL)
}

func TestEmitFrameStampsReceiveTime(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{})
	v.out = make(chan Frame, 1)

	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xD9}
	before := time.Now()
	v.emitFrame(jpeg)
	after := time.Now()

	select {
	case f := <-v.out:
		require.Equal(t, jpeg, f.JPEG)
		require.False(t, f.Timestamp.Before(before), "timestamp must not precede the call")
		require.False(t, f.Timestamp.After(after), "timestamp must not follow the call")
	default:
		t.Fatal("expected emitFrame to enqueue a frame")
	}
}
