package arm_g1

import (
	"context"
	"fmt"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	customAPIID       = 9001
	sportRequestTopic = "api/sport/request"
)

// customActionMap translates ArmAction enum values to the action name sent to the ROS2 node.
var customActionMap = map[string]string{
	"shake_hand":  "shake_hand",
	"face_wave":   "face_wave",
	"hands_up":    "hands_up",
	"stand_still": "stand_still",
	"wave":        "face_wave",
	"show_hand":   "show_hand",
}

type ArmAction string

func (ArmAction) EnumValues() []string {
	return []string{
		"idle",
		"shake_hand",
		"face_wave",
		"hands_up",
		"stand_still",
		"show_hand",
		"wave",
	}
}

type ArmInput struct {
	Action ArmAction `json:"action" description:"The arm gesture to perform"`
}

func init() {
	actions.RegisterInterface(
		"arm_g1",
		"Action interface for Unitree G1 arm gesture control. "+
			"Publishes custom arm motion commands to the g1_arm_action ROS2 node via Zenoh. "+
			"Supported gestures: shake_hand, face_wave, hands_up, stand_still, show_hand, wave.",
		ArmInput{},
	)
	actions.Register("arm_g1/zenoh", newZenohConnector)
}

type zenohConnector struct {
	log       *zap.Logger
	session   *zenohsession.Session
	publisher *zenohsession.Publisher
}

func newZenohConnector(cfg map[string]any) (actions.Connector, error) {
	log := logger.Get()

	var endpoint string
	if ep, ok := cfg["zenoh_endpoint"].(string); ok {
		endpoint = ep
	}

	sess, err := zenohsession.Open(endpoint)
	if err != nil {
		log.Warn("arm_g1/zenoh: zenoh unavailable, arm gestures disabled", zap.Error(err))
		return &zenohConnector{log: log}, nil
	}

	pub, err := sess.DeclarePublisher(sportRequestTopic)
	if err != nil {
		sess.Close()
		log.Warn("arm_g1/zenoh: failed to declare publisher, arm gestures disabled", zap.Error(err))
		return &zenohConnector{log: log}, nil
	}

	log.Info("arm_g1/zenoh: zenoh session opened")
	return &zenohConnector{log: log, session: sess, publisher: pub}, nil
}

func (z *zenohConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("arm_g1/zenoh: unexpected input type %T", input)
	}
	action, _ := args["action"].(string)
	if action == "" || action == "idle" {
		return nil, nil
	}

	actionName, found := customActionMap[action]
	if !found {
		z.log.Warn("arm_g1/zenoh: unknown action, skipping", zap.String("action", action))
		return nil, nil
	}

	if z.publisher == nil {
		return nil, nil
	}

	parameter := fmt.Sprintf(`{"action": "%s"}`, actionName)
	payload := serializeUnitreeRequest(customAPIID, parameter)
	if err := z.publisher.Put(payload); err != nil {
		z.log.Error("arm_g1/zenoh: put failed", zap.Error(err))
		return nil, err
	}

	z.log.Info("arm_g1/zenoh: published", zap.String("action", action), zap.String("mapped", actionName))
	return nil, nil
}

func (z *zenohConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

func (z *zenohConnector) Stop() {
	if z.publisher != nil {
		z.publisher.Drop()
		z.publisher = nil
		z.log.Info("arm_g1/zenoh: publisher dropped")
	}

	if z.session != nil {
		z.session.Close()
		z.session = nil
		z.log.Info("arm_g1/zenoh: zenoh session closed")
	}

}

// serializeUnitreeRequest encodes a Unitree API request in CDR little-endian format.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  identity.id     int64 LE = 0
//	[12] identity.api_id int64 LE
//	[20] lease.id        int64 LE = 0
//	[28] policy.priority int32 LE = 0
//	[32] policy.noreply  bool    = 0
//	[33] padding (3 bytes, align uint32 to data offset 32)
//	[36] parameter length uint32 LE (includes null terminator)
//	[40] parameter bytes  (null-terminated UTF-8)
//	[..] padding to 4-byte (data) boundary
//	[..] binary length    uint32 LE = 0
func serializeUnitreeRequest(apiID int64, parameter string) []byte {
	paramBytes := append([]byte(parameter), 0x00)
	paramLen := uint32(len(paramBytes))

	buf := make([]byte, 0, 76+len(paramBytes))

	// CDR encapsulation header (little-endian)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)

	// identity.id = 0 (data offset 0, already 8-byte aligned)
	buf = zenohsession.AppendInt64LE(buf, 0)

	// identity.api_id
	buf = zenohsession.AppendInt64LE(buf, apiID)

	// lease.id = 0
	buf = zenohsession.AppendInt64LE(buf, 0)

	// policy.priority = 0
	buf = zenohsession.AppendUint32LE(buf, 0)

	// policy.noreply = false
	buf = append(buf, 0x00)

	// 3 bytes padding to align uint32 (parameter length) to data offset 32
	buf = append(buf, 0x00, 0x00, 0x00)

	// parameter length (including null terminator)
	buf = zenohsession.AppendUint32LE(buf, paramLen)

	// parameter string bytes
	buf = append(buf, paramBytes...)

	// padding to 4-byte (data) boundary for binary sequence length
	dataLen := len(buf) - 4 // subtract 4-byte CDR header
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	// binary sequence length = 0
	buf = zenohsession.AppendUint32LE(buf, 0)

	return buf
}
