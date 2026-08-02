package home_assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// capturedPublish records the last publish call made through c.publish.
type capturedPublish struct {
	topic   string
	payload map[string]any
}

func buildMQTTConnector(t *testing.T, broker string) (*MQTTConnector, *capturedPublish) {
	t.Helper()
	conn, err := NewMQTTConnector(map[string]any{
		"broker": broker,
	})
	require.NoError(t, err)
	c, ok := conn.(*MQTTConnector)
	require.True(t, ok, "expected *MQTTConnector")

	captured := &capturedPublish{}
	c.publish = func(topic string, payload []byte) error {
		captured.topic = topic
		var body map[string]any
		_ = json.Unmarshal(payload, &body)
		captured.payload = body
		return nil
	}
	return c, captured
}

func buildFailingMQTTConnector(t *testing.T, broker string, failErr error) *MQTTConnector {
	t.Helper()
	conn, err := NewMQTTConnector(map[string]any{"broker": broker})
	require.NoError(t, err)
	c := conn.(*MQTTConnector)
	c.publish = func(topic string, payload []byte) error {
		return failErr
	}
	return c
}

func TestMQTTConnectorConfig(t *testing.T) {
	t.Run("default values", func(t *testing.T) {
		conn, err := NewMQTTConnector(map[string]any{"broker": "test-broker"})
		require.NoError(t, err)
		c := conn.(*MQTTConnector)
		require.Equal(t, defaultMQTTPort, c.port)
		require.Equal(t, defaultMQTTTopicPrefix, c.topicPrefix)
		require.Equal(t, defaultMQTTTimeout, c.timeout)
	})

	t.Run("custom values", func(t *testing.T) {
		conn, err := NewMQTTConnector(map[string]any{
			"broker":       "test-broker",
			"port":         8883,
			"username":     "user",
			"password":     "pass",
			"topic_prefix": "custom/",
			"timeout":      5.0,
		})
		require.NoError(t, err)
		c := conn.(*MQTTConnector)
		require.Equal(t, 8883, c.port)
		require.Equal(t, "user", c.username)
		require.Equal(t, "pass", c.password)
		require.Equal(t, "custom", c.topicPrefix)
	})

	t.Run("warns but does not fail when broker missing", func(t *testing.T) {
		conn, err := NewMQTTConnector(map[string]any{})
		require.NoError(t, err)
		require.NotNil(t, conn)
	})
}

func TestBuildTopic(t *testing.T) {
	conn, err := NewMQTTConnector(map[string]any{"broker": "test-broker"})
	require.NoError(t, err)
	c := conn.(*MQTTConnector)

	require.Equal(t, "homeassistant/light/bed_light/set", c.buildTopic("light", "light.bed_light"))
	require.Equal(t, "homeassistant/switch/plug/set", c.buildTopic("switch", "plug"))

	conn2, err := NewMQTTConnector(map[string]any{"broker": "test-broker", "topic_prefix": "custom/"})
	require.NoError(t, err)
	c2 := conn2.(*MQTTConnector)
	require.Equal(t, "custom/climate/thermostat/set", c2.buildTopic("climate", "climate.thermostat"))
}

func TestMQTTConnectLight(t *testing.T) {
	t.Run("turn_on", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "homeassistant/light/bed_light/set", captured.topic)
		require.Equal(t, "ON", captured.payload["state"])
	})

	t.Run("turn_off", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "turn_off",
		})
		require.NoError(t, err)
		require.Equal(t, "OFF", captured.payload["state"])
	})

	t.Run("set_brightness", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_brightness",
			"brightness":  "128",
		})
		require.NoError(t, err)
		require.InDelta(t, 128, captured.payload["brightness"], 0.01)
		require.Equal(t, "ON", captured.payload["state"])
	})

	t.Run("set_color", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_color",
			"color":       "green",
		})
		require.NoError(t, err)
		hs, ok := captured.payload["hs_color"].([]any)
		require.True(t, ok)
		require.EqualValues(t, 120, hs[0])
		require.EqualValues(t, 100, hs[1])
	})

	t.Run("unknown color defaults to white", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "set_color",
			"color":       "nonexistent",
		})
		require.NoError(t, err)
		hs, ok := captured.payload["hs_color"].([]any)
		require.True(t, ok)
		require.EqualValues(t, 0, hs[0])
		require.EqualValues(t, 0, hs[1])
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "light.bed_light",
			"device_type": "light",
			"action":      "dance",
		})
		require.NoError(t, err)
		require.Empty(t, captured.topic)
	})
}

func TestMQTTConnectSwitch(t *testing.T) {
	t.Run("turn_on", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "switch.plug",
			"device_type": "switch",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "homeassistant/switch/plug/set", captured.topic)
		require.Equal(t, "ON", captured.payload["state"])
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "switch.plug",
			"device_type": "switch",
			"action":      "set_brightness",
		})
		require.NoError(t, err)
		require.Empty(t, captured.topic)
	})
}

func TestMQTTConnectClimate(t *testing.T) {
	t.Run("set_temperature", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.thermostat",
			"device_type": "climate",
			"action":      "set_temperature",
			"temperature": "24.5",
		})
		require.NoError(t, err)
		require.InDelta(t, 24.5, captured.payload["temperature"], 0.01)
	})

	t.Run("set_temperature empty string falls back to MQTT default 20.0", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.thermostat",
			"device_type": "climate",
			"action":      "set_temperature",
			"temperature": "",
		})
		require.NoError(t, err)
		require.InDelta(t, defaultMQTTTemperature, captured.payload["temperature"], 0.01)
	})

	t.Run("turn_on", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.thermostat",
			"device_type": "climate",
			"action":      "turn_on",
		})
		require.NoError(t, err)
		require.Equal(t, "ON", captured.payload["state"])
	})

	t.Run("unsupported action warns and no-ops", func(t *testing.T) {
		c, captured := buildMQTTConnector(t, "test-broker")
		_, err := c.Connect(context.Background(), map[string]any{
			"entity_id":   "climate.thermostat",
			"device_type": "climate",
			"action":      "cool_mode",
		})
		require.NoError(t, err)
		require.Empty(t, captured.topic)
	})
}

func TestMQTTConnectUnsupportedDeviceType(t *testing.T) {
	c, captured := buildMQTTConnector(t, "test-broker")
	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "vacuum.robot",
		"device_type": "vacuum",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Empty(t, captured.topic)
}

func TestMQTTConnectMissingEntityID(t *testing.T) {
	c, captured := buildMQTTConnector(t, "test-broker")
	_, err := c.Connect(context.Background(), map[string]any{
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err)
	require.Empty(t, captured.topic)
}

func TestMQTTPublishError(t *testing.T) {
	c := buildFailingMQTTConnector(t, "test-broker", fmt.Errorf("mqtt error"))
	_, err := c.Connect(context.Background(), map[string]any{
		"entity_id":   "light.bed_light",
		"device_type": "light",
		"action":      "turn_on",
	})
	require.NoError(t, err) // Connect never returns an error; failures are logged
}

func TestMQTTPublishNoBroker(t *testing.T) {
	conn, err := NewMQTTConnector(map[string]any{})
	require.NoError(t, err)
	c := conn.(*MQTTConnector)

	err = c.publish("some/topic", []byte(`{}`))
	require.Error(t, err)
}

func TestMQTTConnectorInvalidInputType(t *testing.T) {
	c, _ := buildMQTTConnector(t, "test-broker")
	_, err := c.Connect(context.Background(), "not-a-map")
	require.Error(t, err)
}

func TestMQTTConnectorTickAndStop(t *testing.T) {
	c, _ := buildMQTTConnector(t, "test-broker")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.Tick(ctx)

	c.Stop()
}
