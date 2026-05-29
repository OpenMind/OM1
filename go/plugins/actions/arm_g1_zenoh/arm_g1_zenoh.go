package arm_g1_zenoh

import (
	"context"
	"encoding/binary"
	"encoding/json"
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

// ArmAction is the enum of supported arm gestures.
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

// ArmInput is the LLM-facing input struct for the arm action.
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
	log     *zap.Logger
	session *zenohsession.Session
}

func newZenohConnector(cfg map[string]any) (actions.Connector, error) {
	log := logger.Get()

	sess, err := zenohsession.Open()
	if err != nil {
		log.Error("arm_g1/zenoh: failed to open zenoh session", zap.Error(err))
		return &zenohConnector{log: log, session: nil}, nil
	}
	log.Info("arm_g1/zenoh: zenoh session opened")

	return &zenohConnector{log: log, session: sess}, nil
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

	if z.session == nil {
		z.log.Error("arm_g1/zenoh: no zenoh session available")
		return nil, nil
	}

	parameter, err := json.Marshal(map[string]string{"action": actionName})
	if err != nil {
		return nil, fmt.Errorf("arm_g1/zenoh: marshal parameter: %w", err)
	}

	payload := serializeUnitreeRequest(customAPIID, string(parameter))
	if err := z.session.Put(sportRequestTopic, payload); err != nil {
		z.log.Error("arm_g1/zenoh: put failed", zap.Error(err))
		return nil, err
	}

	z.log.Info("arm_g1/zenoh: published", zap.String("action", action), zap.String("mapped", actionName))
	return nil, nil
}

func (z *zenohConnector) Tick(_ context.Context) {}

func (z *zenohConnector) Stop() {
	if z.session != nil {
		z.session.Close()
		z.session = nil
		z.log.Info("arm_g1/zenoh: zenoh session closed")
	}
}

// serializeUnitreeRequest encodes a Unitree API request in CDR little-endian format.
//
// Wire layout (offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  padding (align int64 to offset 8)
//	[8]  identity.id     int64 LE = 0
//	[16] identity.api_id int64 LE
//	[24] lease.id        int64 LE = 0
//	[32] policy.priority int32 LE = 0
//	[36] policy.noreply  bool    = 0
//	[37] padding (align uint32 to offset 40)
//	[40] parameter length uint32 LE (includes null terminator)
//	[44] parameter bytes  (null-terminated UTF-8)
//	[..] padding to 4-byte boundary
//	[..] binary length    uint32 LE = 0
func serializeUnitreeRequest(apiID int64, parameter string) []byte {
	paramBytes := append([]byte(parameter), 0x00) // null-terminated
	paramLen := uint32(len(paramBytes))

	buf := make([]byte, 0, 80+len(paramBytes))

	// CDR encapsulation header (little-endian)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)

	// 4 bytes padding to align first int64 to offset 8
	buf = append(buf, 0x00, 0x00, 0x00, 0x00)

	// identity.id = 0
	buf = appendInt64LE(buf, 0)

	// identity.api_id
	buf = appendInt64LE(buf, apiID)

	// lease.id = 0
	buf = appendInt64LE(buf, 0)

	// policy.priority = 0
	buf = appendUint32LE(buf, 0)

	// policy.noreply = false
	buf = append(buf, 0x00)

	// 3 bytes padding to align uint32 (parameter length) to offset 40
	buf = append(buf, 0x00, 0x00, 0x00)

	// parameter length (including null terminator)
	buf = appendUint32LE(buf, paramLen)

	// parameter string bytes
	buf = append(buf, paramBytes...)

	// padding to 4-byte boundary for binary sequence length
	if pad := (4 - len(buf)%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	// binary sequence length = 0
	buf = appendUint32LE(buf, 0)

	return buf
}

func appendInt64LE(buf []byte, v int64) []byte {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], uint64(v))
	return append(buf, b[:]...)
}

func appendUint32LE(buf []byte, v uint32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return append(buf, b[:]...)
}
