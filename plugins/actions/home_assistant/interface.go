// Package home_assistant provides an action plugin that controls smart home
// devices (lights, switches, climate/thermostats) via Home Assistant, using a
// configurable connector: REST API, WebSocket, or MQTT.
package home_assistant

import (
	"github.com/openmind/om1/internal/actions"
)

// HAAction is the LLM-facing enum for the operation to perform on a device.
type HAAction string

// EnumValues lists the actions the LLM may request.
func (HAAction) EnumValues() []string {
	return []string{
		"turn_on",
		"turn_off",
		"set_brightness",
		"set_color",
		"set_temperature",
	}
}

// HADeviceType is the LLM-facing enum for the kind of device being controlled.
type HADeviceType string

// EnumValues lists the device types supported by the connectors.
func (HADeviceType) EnumValues() []string {
	return []string{
		"light",
		"switch",
		"climate",
	}
}

// COLOR_MAP maps friendly color names to Home Assistant HS (hue, saturation)
// values, shared by all connectors (rest, websocket, mqtt).
var COLOR_MAP = map[string][2]int{
	"red":        {0, 100},
	"green":      {120, 100},
	"blue":       {240, 100},
	"yellow":     {60, 100},
	"orange":     {30, 100},
	"purple":     {270, 100},
	"pink":       {300, 100},
	"white":      {0, 0},
	"warm white": {30, 20},
	"cool white": {200, 10},
	"cyan":       {180, 100},
}

// HomeAssistantInput is the input the LLM provides when calling this action.
//
// Brightness and Temperature are strings (not int/float) because the LLM may
// emit empty-string values for optional numeric fields when they are not
// relevant to the chosen action (e.g. turn_on). Each connector parses these
// safely and falls back to sane defaults on empty/invalid input.
//
// Brightness is a direct 0-255 value (not a percentage), matching the
// original Home Assistant API convention.
type HomeAssistantInput struct {
	EntityID    string       `json:"entity_id" description:"The Home Assistant entity ID to control, e.g. 'light.bed_light' or 'switch.living_room'."`
	DeviceType  HADeviceType `json:"device_type" description:"The type of device being controlled."`
	Action      HAAction     `json:"action" description:"The operation to perform on the device."`
	Brightness  string       `json:"brightness" description:"Brightness value (0-255) as a string, used only with set_brightness. Leave empty for other actions."`
	Temperature string       `json:"temperature" description:"Target temperature (Celsius) as a string, used only with set_temperature. Leave empty for other actions."`
	Color       string       `json:"color" description:"Color name (e.g. 'red', 'blue', 'warm white'), used only with set_color. Leave empty for other actions."`
}

func init() {
	actions.RegisterInterface(
		"home_assistant",
		"Action interface that controls smart home devices (lights, switches, "+
			"climate/thermostats) through Home Assistant. Use this when the user "+
			"asks to control a smart home device, e.g. \"turn on the bed light\" "+
			"-> device_type=\"light\", action=\"turn_on\", entity_id=\"light.bed_light\".",
		HomeAssistantInput{},
	)
}
