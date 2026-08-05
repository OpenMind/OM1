package connector

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGetAQILevel_Good(t *testing.T) {
	label, desc := GetAQILevel(25)
	require.Equal(t, "GOOD", label)
	require.Contains(t, desc, "satisfactory")
}

func TestGetAQILevel_Moderate(t *testing.T) {
	label, _ := GetAQILevel(75)
	require.Equal(t, "MODERATE", label)
}

func TestGetAQILevel_UnhealthySensitive(t *testing.T) {
	label, _ := GetAQILevel(120)
	require.Equal(t, "UNHEALTHY FOR SENSITIVE GROUPS", label)
}

func TestGetAQILevel_Unhealthy(t *testing.T) {
	label, _ := GetAQILevel(180)
	require.Equal(t, "UNHEALTHY", label)
}

func TestGetAQILevel_VeryUnhealthy(t *testing.T) {
	label, _ := GetAQILevel(250)
	require.Equal(t, "VERY UNHEALTHY", label)
}

func TestGetAQILevel_Hazardous(t *testing.T) {
	label, _ := GetAQILevel(400)
	require.Equal(t, "HAZARDOUS", label)
}

func TestGetAQILevel_Boundaries(t *testing.T) {
	// Exact threshold values must fall in the lower (inclusive) tier,
	// mirroring Python's `if aqi <= threshold`.
	label, _ := GetAQILevel(50)
	require.Equal(t, "GOOD", label)

	label, _ = GetAQILevel(51)
	require.Equal(t, "MODERATE", label)

	label, _ = GetAQILevel(150)
	require.Equal(t, "UNHEALTHY FOR SENSITIVE GROUPS", label)

	label, _ = GetAQILevel(151)
	require.Equal(t, "UNHEALTHY", label)
}

func TestNewAirQualityData_Defaults(t *testing.T) {
	d := NewAirQualityData()
	require.Equal(t, "Unknown", d.Location)
	require.Equal(t, "Unknown", d.Source)
	require.Nil(t, d.AQI)
	require.Nil(t, d.PM25)
}
