package actions

import (
	"reflect"
	"strings"
)

// Enumer is an interface for input fields that should be treated as enums.
type Enumer interface {
	EnumValues() []string
}

// InterfaceSpec holds the description and input example for an action interface
type InterfaceSpec struct {
	description  string
	inputExample any
}

var InterfaceRegistry = map[string]InterfaceSpec{}

// RegisterInterface registers an action interface with a name, description, and input example struct.
func RegisterInterface(actionName, description string, inputExample any) {
	InterfaceRegistry[actionName] = InterfaceSpec{
		description:  description,
		inputExample: inputExample,
	}
}

// BuildSchemaForAction looks up the input struct for the given actionName and llmLabel.
func BuildSchemaForAction(actionName, llmLabel string) (map[string]any, bool) {
	spec, ok := InterfaceRegistry[actionName]
	if !ok {
		return nil, false
	}

	return BuildSchema(llmLabel, spec.description, spec.inputExample), true
}

// BuildSchema generates an OpenAI function schema by reflecting over the fields
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
func BuildSchema(llmLabel, description string, inputExample any) map[string]any {
	inputType := reflect.TypeOf(inputExample)
	if inputType.Kind() == reflect.Ptr {
		inputType = inputType.Elem()
	}

	properties := map[string]any{}
	required := []string{}

	for i := 0; i < inputType.NumField(); i++ {
		field := inputType.Field(i)

		propertyName := field.Tag.Get("json")
		if propertyName == "" {
			propertyName = strings.ToLower(field.Name)
		}
		propertyName = strings.Split(propertyName, ",")[0]

		fieldDescription := field.Tag.Get("description")
		if fieldDescription == "" {
			fieldDescription = "The " + propertyName + " parameter"
		}

		properties[propertyName] = BuildPropertySchema(field.Type, fieldDescription)
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

// BuildPropertySchema returns the JSON-schema fragment for one input field.
func BuildPropertySchema(fieldType reflect.Type, description string) map[string]any {
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
		"type":        KindToJSONType(fieldType.Kind()),
		"description": description,
	}
}

// KindToJSONType maps a Go reflect.Kind to the corresponding JSON Schema type.
func KindToJSONType(kind reflect.Kind) string {
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
