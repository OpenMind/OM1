package luma

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewFaceSizeWatch_Defaults(t *testing.T) {
	bg, err := NewFaceSizeWatch(map[string]any{})
	require.NoError(t, err)
	require.NotNil(t, bg)

	fsw := bg.(*FaceSizeWatch)
	require.Equal(t, float64(3000), fsw.minArea)
	require.NotNil(t, fsw.provider)
	require.NotNil(t, fsw.log)
}

func TestNewFaceSizeWatch_CustomConfig(t *testing.T) {
	bg, err := NewFaceSizeWatch(map[string]any{
		"face_http_base_url":     "http://localhost:9999",
		"face_recent_sec":        2.0,
		"face_poll_interval_sec": 1.0,
		"min_face_area":          5000.0,
	})
	require.NoError(t, err)

	fsw := bg.(*FaceSizeWatch)
	require.Equal(t, float64(5000), fsw.minArea)
}

func TestFaceSizeWatch_Stop(t *testing.T) {
	bg, err := NewFaceSizeWatch(map[string]any{})
	require.NoError(t, err)
	bg.(*FaceSizeWatch).Stop()
}
