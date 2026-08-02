package home_assistant

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
)

const (
	defaultMQTTPort        = 1883
	defaultMQTTTopicPrefix = "homeassistant"
	defaultMQTTTimeout     = 10 * time.Second
	defaultMQTTTemperature = 20.0
)

// MQTTConfig is the decoded plugin configuration for the MQTT connector.
type MQTTConfig struct {
	// Broker is the MQTT broker hostname or IP address.
	Broker string `json:"broker"`

	// Port is the MQTT broker port. 0 means use the default (1883).
	Port int `json:"port"`

	// Username is the MQTT broker username (optional).
	Username string `json:"username"`

	// Password is the MQTT broker password (optional).
	Password string `json:"password"`

	// TopicPrefix is the base topic prefix for Home Assistant.
	// Trailing slashes are ignored. Empty means use the default ("homeassistant").
	TopicPrefix string `json:"topic_prefix"`

	// Timeout is the connection timeout in seconds. 0 means use the default.
	Timeout float64 `json:"timeout"`
}

// mqttPublishFunc publishes a single JSON payload to a topic and reports
// success. It is a field on Connector so tests can substitute a fake
// implementation without a real MQTT broker.
type mqttPublishFunc func(topic string, payload []byte) error

// MQTTConnector implements actions.Connector by publishing device control
// commands to MQTT topics following the Home Assistant convention:
// <prefix>/<domain>/<entity>/set
type MQTTConnector struct {
	log         *zap.Logger
	broker      string
	port        int
	username    string
	password    string
	topicPrefix string
	timeout     time.Duration

	publish mqttPublishFunc
}

func init() {
	actions.Register("home_assistant/mqtt", NewMQTTConnector)
}

// NewMQTTConnector builds the connector from a decoded config map.
func NewMQTTConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg MQTTConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("home_assistant/mqtt")

	if cfg.Broker == "" {
		log.Warn("broker not provided")
	}

	port := cfg.Port
	if port == 0 {
		port = defaultMQTTPort
	}

	prefix := cfg.TopicPrefix
	if prefix == "" {
		prefix = defaultMQTTTopicPrefix
	}
	prefix = strings.TrimRight(prefix, "/")

	timeout := defaultMQTTTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout * float64(time.Second))
	}

	c := &MQTTConnector{
		log:         log,
		broker:      cfg.Broker,
		port:        port,
		username:    cfg.Username,
		password:    cfg.Password,
		topicPrefix: prefix,
		timeout:     timeout,
	}
	c.publish = c.publishMQTT

	return c, nil
}

// buildTopic builds the MQTT command topic for an entity, following the
// Home Assistant convention <prefix>/<domain>/<entity>/set. If entityID
// contains a dot (e.g. "light.bed_light"), only the part after the dot is
// used as the entity name.
func (c *MQTTConnector) buildTopic(domain, entityID string) string {
	name := entityID
	if idx := strings.Index(entityID, "."); idx != -1 {
		name = entityID[idx+1:]
	}
	return fmt.Sprintf("%s/%s/%s/set", c.topicPrefix, domain, name)
}

// publishMQTT is the real (non-test) implementation of mqttPublishFunc. It
// opens a short-lived MQTT connection, publishes one message, and
// disconnects, mirroring the ephemeral-client pattern of the original
// Python implementation.
func (c *MQTTConnector) publishMQTT(topic string, payload []byte) error {
	if c.broker == "" {
		return fmt.Errorf("home_assistant/mqtt: broker not set")
	}

	opts := mqtt.NewClientOptions()
	opts.AddBroker(fmt.Sprintf("tcp://%s:%d", c.broker, c.port))
	if c.username != "" {
		opts.SetUsername(c.username)
	}
	if c.password != "" {
		opts.SetPassword(c.password)
	}
	opts.SetConnectTimeout(c.timeout)

	client := mqtt.NewClient(opts)

	token := client.Connect()
	if !token.WaitTimeout(c.timeout) {
		return fmt.Errorf("home_assistant/mqtt: connection timed out")
	}
	if err := token.Error(); err != nil {
		return fmt.Errorf("home_assistant/mqtt: connect error: %w", err)
	}
	defer client.Disconnect(250)

	pubToken := client.Publish(topic, 0, false, payload)
	if !pubToken.WaitTimeout(c.timeout) {
		return fmt.Errorf("home_assistant/mqtt: publish timed out")
	}
	return pubToken.Error()
}

// doPublish marshals payload to JSON and publishes it via c.publish,
// logging the outcome.
func (c *MQTTConnector) doPublish(topic string, payload map[string]any) {
	data, err := json.Marshal(payload)
	if err != nil {
		c.log.Error("marshal payload", zap.Error(err))
		return
	}

	if err := c.publish(topic, data); err != nil {
		c.log.Error("MQTT error", zap.String("topic", topic), zap.Error(err))
		return
	}

	c.log.Info("published", zap.String("topic", topic), zap.ByteString("payload", data))
}

// Connect resolves the requested device action and publishes the
// corresponding MQTT command.
func (c *MQTTConnector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("home_assistant/mqtt: unexpected input type %T", input)
	}

	entityID, _ := args["entity_id"].(string)
	deviceType, _ := args["device_type"].(string)
	action, _ := args["action"].(string)
	brightness, _ := args["brightness"].(string)
	temperature, _ := args["temperature"].(string)
	color, _ := args["color"].(string)

	if entityID == "" {
		return nil, nil
	}

	switch deviceType {
	case "light":
		c.connectLight(entityID, action, brightness, color)
	case "switch":
		c.connectSwitch(entityID, action)
	case "climate":
		c.connectClimate(entityID, action, temperature)
	default:
		c.log.Warn("device_type not supported", zap.String("device_type", deviceType))
	}

	return nil, nil
}

func (c *MQTTConnector) connectLight(entityID, action, brightness, color string) {
	topic := c.buildTopic("light", entityID)
	switch action {
	case "turn_on":
		c.doPublish(topic, map[string]any{"state": "ON"})
	case "turn_off":
		c.doPublish(topic, map[string]any{"state": "OFF"})
	case "set_brightness":
		v := defaultRESTBrightness // reuse 255 default constant, brightness is 0-255 direct value
		if brightness != "" {
			if parsed, err := strconv.Atoi(strings.TrimSpace(brightness)); err == nil {
				v = parsed
			}
		}
		c.doPublish(topic, map[string]any{"state": "ON", "brightness": v})
	case "set_color":
		hs := resolveHSColor(color)
		c.doPublish(topic, map[string]any{"state": "ON", "hs_color": []int{hs[0], hs[1]}})
	default:
		c.log.Warn("action not supported for light", zap.String("action", action))
	}
}

func (c *MQTTConnector) connectSwitch(entityID, action string) {
	topic := c.buildTopic("switch", entityID)
	switch action {
	case "turn_on":
		c.doPublish(topic, map[string]any{"state": "ON"})
	case "turn_off":
		c.doPublish(topic, map[string]any{"state": "OFF"})
	default:
		c.log.Warn("action not supported for switch", zap.String("action", action))
	}
}

func (c *MQTTConnector) connectClimate(entityID, action, temperature string) {
	topic := c.buildTopic("climate", entityID)
	switch action {
	case "turn_on":
		c.doPublish(topic, map[string]any{"state": "ON"})
	case "turn_off":
		c.doPublish(topic, map[string]any{"state": "OFF"})
	case "set_temperature":
		t := parseTemperature(temperature, defaultMQTTTemperature)
		c.doPublish(topic, map[string]any{"temperature": t})
	default:
		c.log.Warn("action not supported for climate", zap.String("action", action))
	}
}

// Tick blocks until ctx is cancelled; this connector is event-driven.
func (c *MQTTConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since each Connect call manages its own connection lifecycle.
func (c *MQTTConnector) Stop() {}
