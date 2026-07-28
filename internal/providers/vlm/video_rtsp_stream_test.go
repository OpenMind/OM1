package vlm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestVideoRTSPDefaultURLValue pins the actual default endpoint string to the
// video-processor's clean (pre-CV) camera view. The sibling
// TestNewVideoRTSPStreamDefaults asserts against the defaultRTSPURL constant, so
// this guards the constant's value itself against accidental changes.
func TestVideoRTSPDefaultURLValue(t *testing.T) {
	require.Equal(t, "rtsp://localhost:8556/raw", defaultRTSPURL)

	v := NewVideoRTSPStream(VideoRTSPStreamConfig{})
	require.Equal(t, "rtsp://localhost:8556/raw", v.cfg.RTSPURL)
}

// TestNewVideoRTSPStreamPreservesExplicitURL covers the non-default branch: a
// caller-supplied URL must not be overwritten.
func TestNewVideoRTSPStreamPreservesExplicitURL(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{RTSPURL: "rtsp://example.test/live"})
	require.Equal(t, "rtsp://example.test/live", v.cfg.RTSPURL)
}
