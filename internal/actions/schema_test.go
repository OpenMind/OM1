package actions

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

type moveInput struct {
	Direction directionEnum `json:"direction" description:"which way to move"`
	Distance  int           `json:"distance"`
	Speed     float64       `json:"speed"`
	Urgent    bool          `json:"urgent"`
	internal  string
}

type directionEnum string

func (directionEnum) EnumValues() []string { return []string{"left", "right"} }

func TestKindToJSONType(t *testing.T) {
	require.Equal(t, "integer", KindToJSONType(reflect.Int))
	require.Equal(t, "integer", KindToJSONType(reflect.Uint32))
	require.Equal(t, "number", KindToJSONType(reflect.Float64))
	require.Equal(t, "boolean", KindToJSONType(reflect.Bool))
	require.Equal(t, "string", KindToJSONType(reflect.String))
	require.Equal(t, "string", KindToJSONType(reflect.Slice), "unknown kinds fall back to string")
}

func TestBuildPropertySchemaEnum(t *testing.T) {
	got := BuildPropertySchema(reflect.TypeOf(directionEnum("")), "dir")
	require.Equal(t, "string", got["type"])
	require.Equal(t, []string{"left", "right"}, got["enum"])
	require.Equal(t, "dir", got["description"])
}

func TestBuildPropertySchemaScalar(t *testing.T) {
	got := BuildPropertySchema(reflect.TypeOf(0), "count")
	require.Equal(t, "integer", got["type"])
	require.NotContains(t, got, "enum")
}

func TestBuildSchema(t *testing.T) {
	schema := BuildSchema("move", "Move the robot", moveInput{})
	require.Equal(t, "function", schema["type"])

	fn := schema["function"].(map[string]any)
	require.Equal(t, "move", fn["name"])
	require.Equal(t, "Move the robot", fn["description"])
	require.Equal(t, true, fn["strict"])

	params := fn["parameters"].(map[string]any)
	require.Equal(t, "object", params["type"])
	require.Equal(t, false, params["additionalProperties"])

	props := params["properties"].(map[string]any)
	require.Equal(t, "which way to move", props["direction"].(map[string]any)["description"], "description tag wins")
	require.Equal(t, "integer", props["distance"].(map[string]any)["type"])
	require.Equal(t, "number", props["speed"].(map[string]any)["type"])
	require.Equal(t, "boolean", props["urgent"].(map[string]any)["type"])
	require.Equal(t, "The internal parameter", props["internal"].(map[string]any)["description"], "auto description for untagged field")

	required := params["required"].([]string)
	require.ElementsMatch(t, []string{"direction", "distance", "speed", "urgent", "internal"}, required)
}

func TestBuildSchemaHandlesPointer(t *testing.T) {
	schema := BuildSchema("move", "desc", &moveInput{})
	require.NotNil(t, schema["function"])
}

func TestRegisterInterfaceAndBuildForAction(t *testing.T) {
	RegisterInterface("test_move", "Move action", moveInput{})
	t.Cleanup(func() { delete(InterfaceRegistry, "test_move") })

	schema, ok := BuildSchemaForAction("test_move", "moveLabel")
	require.True(t, ok)
	require.Equal(t, "moveLabel", schema["function"].(map[string]any)["name"])

	_, ok = BuildSchemaForAction("unregistered", "x")
	require.False(t, ok)
}
