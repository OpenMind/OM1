package go2

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/geometry"
)

func TestParseLocationsObjectForm(t *testing.T) {
	body := `{
		"Kitchen": {"pose": {"position": {"x": 1.5, "y": 2.0, "z": 0.0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}},
		"sofa": {"name": "Living Room Sofa", "pose": {"position": {"x": -3.0, "y": 4.0, "z": 0.0}}}
	}`

	locs, err := parseLocations(strings.NewReader(body))
	require.NoError(t, err)

	kitchen, ok := locs["kitchen"]
	require.True(t, ok, "key should be lowercased")
	assert.Equal(t, "Kitchen", kitchen.Name, "missing name falls back to the map key")
	assert.Equal(t, 1.5, kitchen.Pose.Position.X)
	assert.Equal(t, 1.0, kitchen.Pose.Orientation.W)

	sofa, ok := locs["sofa"]
	require.True(t, ok)
	assert.Equal(t, "Living Room Sofa", sofa.Name, "explicit name is preserved")
	assert.Equal(t, -3.0, sofa.Pose.Position.X)
}

func TestParseLocationsListForm(t *testing.T) {
	body := `[
		{"name": "Garage", "pose": {"position": {"x": 5.0, "y": 0.0, "z": 0.0}}},
		{"name": "", "pose": {}}
	]`

	locs, err := parseLocations(strings.NewReader(body))
	require.NoError(t, err)

	assert.Len(t, locs, 1, "entries without a name are skipped")
	garage, ok := locs["garage"]
	require.True(t, ok)
	assert.Equal(t, 5.0, garage.Pose.Position.X)
}

func TestParseLocationsMessageEnvelope(t *testing.T) {
	// The API may wrap the payload as a JSON string under "message".
	body := `{"message": "{\"kitchen\": {\"pose\": {\"position\": {\"x\": 9.0, \"y\": 0, \"z\": 0}}}}"}`

	locs, err := parseLocations(strings.NewReader(body))
	require.NoError(t, err)

	kitchen, ok := locs["kitchen"]
	require.True(t, ok)
	assert.Equal(t, 9.0, kitchen.Pose.Position.X)
}

func TestGetLocationAndAllNames(t *testing.T) {
	p := NewLocationsProvider("http://unused", "", 0, 0)
	p.locations = map[string]Location{
		"kitchen": {Name: "Kitchen", Pose: geometry.Pose{Position: geometry.Point{X: 1}}},
	}

	_, ok := p.GetLocation("  KITCHEN ")
	assert.True(t, ok, "lookup is case-insensitive and trims whitespace")

	_, ok = p.GetLocation("bedroom")
	assert.False(t, ok)

	_, ok = p.GetLocation("")
	assert.False(t, ok)

	assert.Equal(t, []string{"Kitchen"}, p.AllNames())
}
