package actions

import (
	"reflect"
	"strings"
)

// Enumer is implemented by action input field types that have a fixed set of
// valid string values (i.e. string enums).
//
// Example:
//
//	type MovementAction string
//
//	func (MovementAction) EnumValues() []string {
//	    return []string{"stand still", "sit", "walk", "run"}
//	}
type Enumer interface {
	EnumValues() []string
}

// interfaceSpec holds what a plugin registered for one action name.
type interfaceSpec struct {
	description  string
	inputExample any // zero-value instance of the input struct
}

var interfaceRegistry = map[string]interfaceSpec{}

// RegisterInterface declares the input type and human-readable description for
// an action.  actionName must match the "name" field used in config files
// (e.g. "move").  inputExample should be a zero-value instance of the input
// struct (e.g. MoveInput{}).
//
// Call this once per action name, typically from an init() function alongside
// the connector Register calls.
func RegisterInterface(actionName, description string, inputExample any) {
	interfaceRegistry[actionName] = interfaceSpec{
		description:  description,
		inputExample: inputExample,
	}
}

// BuildSchemaForAction looks up the registered interface for actionName and
// returns the OpenAI-compatible function schema.  Returns nil, false when no
// interface has been registered for that name.
func BuildSchemaForAction(actionName, llmLabel string) (map[string]any, bool) {
	spec, ok := interfaceRegistry[actionName]
	if !ok {
		return nil, false
	}
	return buildSchema(llmLabel, spec.description, spec.inputExample), true
}

// buildSchema generates an OpenAI function schema by reflecting over the fields
// of inputExample.  It mirrors the Python generate_function_schema_from_action:
//   - fields whose type implements Enumer → {"type":"string","enum":[...]}
//   - string  → {"type":"string"}
//   - int     → {"type":"integer"}
//   - float64 → {"type":"number"}
//   - bool    → {"type":"boolean"}
//   - other   → {"type":"string"} (safe fallback)
//
// The "json" struct tag is used for the property name; the "description" struct
// tag overrides the auto-generated description.
func buildSchema(llmLabel, description string, inputExample any) map[string]any {
	inputType := reflect.TypeOf(inputExample)
	if inputType.Kind() == reflect.Ptr {
		inputType = inputType.Elem()
	}

	properties := map[string]any{}
	required := []string{}

	for i := 0; i < inputType.NumField(); i++ {
		field := inputType.Field(i)

		// Derive JSON property name from the "json" tag.
		propertyName := field.Tag.Get("json")
		if propertyName == "" {
			propertyName = strings.ToLower(field.Name)
		}
		propertyName = strings.Split(propertyName, ",")[0] // strip omitempty etc.

		// Prefer an explicit "description" tag; fall back to a generated one.
		fieldDescription := field.Tag.Get("description")
		if fieldDescription == "" {
			fieldDescription = "The " + propertyName + " parameter"
		}

		properties[propertyName] = buildPropertySchema(field.Type, fieldDescription)
		required = append(required, propertyName)
	}

	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        llmLabel,
			"description": description,
			"parameters": map[string]any{
				"type":                 "object",
				"properties":           properties,
				"required":             required,
				"additionalProperties": false,
			},
			"strict": true,
		},
	}
}

// buildPropertySchema returns the JSON-schema fragment for one input field.
func buildPropertySchema(fieldType reflect.Type, description string) map[string]any {
	// Check if a zero value of this type implements Enumer.
	zeroValue := reflect.Zero(fieldType)
	if enumer, ok := zeroValue.Interface().(Enumer); ok {
		values := enumer.EnumValues()
		return map[string]any{
			"type":        "string",
			"enum":        values,
			"description": description,
		}
	}

	return map[string]any{
		"type":        kindToJSONType(fieldType.Kind()),
		"description": description,
	}
}

// kindToJSONType maps a Go reflect.Kind to the corresponding JSON Schema type.
func kindToJSONType(kind reflect.Kind) string {
	switch kind {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	case reflect.Bool:
		return "boolean"
	default:
		return "string"
	}
}
