package actions

import (
	"testing"
)

type testMovementAction string

func (testMovementAction) EnumValues() []string { return []string{"walk", "run", "sit"} }

type testMoveInput struct {
	Action   testMovementAction `json:"action" description:"Movement command"`
	Speed    float64            `json:"speed"`
	Repeated bool               `json:"repeated"`
}

type testSpeakInput struct {
	Action string `json:"action" description:"Text to speak"`
}

func TestBuildSchema_EnumField(t *testing.T) {
	schema := buildSchema("move", "Move the robot.", testMoveInput{})

	fn, ok := schema["function"].(map[string]any)
	if !ok {
		t.Fatal("missing 'function' key")
	}
	if fn["name"] != "move" {
		t.Errorf("name = %q, want %q", fn["name"], "move")
	}
	if fn["description"] != "Move the robot." {
		t.Errorf("description = %q", fn["description"])
	}

	params, ok := fn["parameters"].(map[string]any)
	if !ok {
		t.Fatal("missing 'parameters' key")
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("missing 'properties' key")
	}

	actionProp, ok := props["action"].(map[string]any)
	if !ok {
		t.Fatal("missing 'action' property")
	}
	if actionProp["type"] != "string" {
		t.Errorf("action.type = %q, want %q", actionProp["type"], "string")
	}
	enumValues, ok := actionProp["enum"].([]string)
	if !ok {
		t.Fatal("action.enum is not []string")
	}
	if len(enumValues) != 3 || enumValues[0] != "walk" {
		t.Errorf("action.enum = %v", enumValues)
	}
	if actionProp["description"] != "Movement command" {
		t.Errorf("action.description = %q", actionProp["description"])
	}

	speedProp, ok := props["speed"].(map[string]any)
	if !ok {
		t.Fatal("missing 'speed' property")
	}
	if speedProp["type"] != "number" {
		t.Errorf("speed.type = %q, want %q", speedProp["type"], "number")
	}

	repeatedProp, ok := props["repeated"].(map[string]any)
	if !ok {
		t.Fatal("missing 'repeated' property")
	}
	if repeatedProp["type"] != "boolean" {
		t.Errorf("repeated.type = %q, want %q", repeatedProp["type"], "boolean")
	}

	requiredRaw := params["required"]
	required, ok := requiredRaw.([]string)
	if !ok {
		t.Fatalf("required is %T, want []string", requiredRaw)
	}
	if len(required) != 3 {
		t.Errorf("required len = %d, want 3: %v", len(required), required)
	}
}

func TestBuildSchema_StringField(t *testing.T) {
	schema := buildSchema("speak", "Speak text.", testSpeakInput{})

	fn := schema["function"].(map[string]any)
	params := fn["parameters"].(map[string]any)
	props := params["properties"].(map[string]any)

	actionProp, ok := props["action"].(map[string]any)
	if !ok {
		t.Fatal("missing 'action' property")
	}
	if actionProp["type"] != "string" {
		t.Errorf("action.type = %q, want string", actionProp["type"])
	}
	if _, hasEnum := actionProp["enum"]; hasEnum {
		t.Error("plain string field should not have 'enum' key")
	}
	if actionProp["description"] != "Text to speak" {
		t.Errorf("action.description = %q", actionProp["description"])
	}
}

func TestBuildSchemaForAction_Registry(t *testing.T) {
	RegisterInterface("_test_action", "Test description.", testSpeakInput{})

	schema, ok := BuildSchemaForAction("_test_action", "test_action")
	if !ok {
		t.Fatal("BuildSchemaForAction returned false for a registered action")
	}
	fn := schema["function"].(map[string]any)
	if fn["name"] != "test_action" {
		t.Errorf("name = %q, want test_action", fn["name"])
	}

	_, notFound := BuildSchemaForAction("unregistered_action", "x")
	if notFound {
		t.Error("expected false for unregistered action")
	}
}
