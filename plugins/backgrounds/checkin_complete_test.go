package backgrounds

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewCheckinComplete_Defaults(t *testing.T) {
	bg, err := NewCheckinComplete(map[string]any{})
	require.NoError(t, err)
	require.NotNil(t, bg)

	cc := bg.(*CheckinComplete)
	require.Equal(t, float64(3000), cc.minArea)
	require.Equal(t, "QRScannerRTSP", cc.scanIOKey)
	require.False(t, cc.scanSeen)
	require.NotNil(t, cc.face)
	require.NotNil(t, cc.log)
}

func TestNewCheckinComplete_CustomConfig(t *testing.T) {
	bg, err := NewCheckinComplete(map[string]any{
		"face_http_base_url": "http://localhost:9999",
		"min_face_area":      8000.0,
		"poll_interval_sec":  2.0,
		"scan_io_key":        "QRScanner",
		"grace_period_sec":   10.0,
	})
	require.NoError(t, err)

	cc := bg.(*CheckinComplete)
	require.Equal(t, float64(8000), cc.minArea)
	require.Equal(t, "QRScanner", cc.scanIOKey)
}

func TestCheckinComplete_Stop(t *testing.T) {
	bg, err := NewCheckinComplete(map[string]any{})
	require.NoError(t, err)
	bg.(*CheckinComplete).Stop()
}
