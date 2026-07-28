package vlm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestNewVideoRTSPStreamDefaults pins the VLM's default RTSP source to the
// video-processor's clean (pre-CV) camera view (:8556/raw), which gives the best
// scene descriptions, and checks the other zero-value fallbacks are applied.
func TestNewVideoRTSPStreamDefaults(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{})

	require.Equal(t, "rtsp://localhost:8556/raw", v.cfg.RTSPURL)
	require.Equal(t, defaultRTSPWidth, v.cfg.Width)
	require.Equal(t, defaultRTSPHeight, v.cfg.Height)
	require.Equal(t, defaultFPS, v.cfg.FPS)
	require.Equal(t, defaultJPEGQuality, v.cfg.JPEGQuality)
}

func TestNewVideoRTSPStreamPreservesExplicitURL(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{RTSPURL: "rtsp://example.test/live"})
	require.Equal(t, "rtsp://example.test/live", v.cfg.RTSPURL)
}
