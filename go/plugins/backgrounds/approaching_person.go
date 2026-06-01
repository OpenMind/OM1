package backgrounds

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	personGreetingTopic = "om/person_greeting"
	contextUpdateTopic  = "om/mode/context"

	personStatusApproaching  = int8(0)
	personStatusApproached   = int8(1)
	personStatusSwitch       = int8(2)
	personStatusConversation = int8(3)

	approachingPollInterval = 5 * time.Second
)

func init() {
	bg.Register("ApproachingPerson", NewApproachingPerson)
}

// ApproachingPerson is a background task that monitors for approaching people via
// Zenoh messages and updates the mode context accordingly.
type ApproachingPerson struct {
	log      *zap.Logger
	session  *zenohsession.Session
	sub      *zenohsession.Subscriber
	greeting *providers.GreetingConversationStateMachineProvider

	mu                 sync.Mutex
	isPersonApproached bool
}

// NewApproachingPerson constructs the ApproachingPerson background task and
// subscribes to the person-greeting topic.
func NewApproachingPerson(configMap map[string]any) (bg.Background, error) {
	log := logger.Get()

	a := &ApproachingPerson{log: log, greeting: providers.Greeting()}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Error("ApproachingPerson: zenoh unavailable", zap.Error(err))
		return a, nil
	}
	a.session = sess

	sub, err := sess.DeclareSubscriber(personGreetingTopic, a.onPersonGreeting)
	if err != nil {
		log.Error("ApproachingPerson: failed to declare subscriber", zap.Error(err))
	} else {
		a.sub = sub
	}

	log.Info("ApproachingPerson background task initialized.")
	return a, nil
}

// onPersonGreeting is called when a message is received on the person-greeting topic.
func (a *ApproachingPerson) onPersonGreeting(data []byte) {
	a.log.Debug("Person greeting detected via Zenoh message.")

	status, err := deserializePersonGreetingStatus(data)
	if err != nil {
		a.log.Error("ApproachingPerson: failed to deserialize status", zap.Error(err))
		return
	}

	if status != personStatusApproached {
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	a.log.Info("Person is approaching. Triggering greeting mode.")
	a.isPersonApproached = true

	a.updateContext(map[string]any{"approaching_detected": true})

	a.greeting.ResetState(providers.StateEngaging)
}

// updateContext publishes the given key/value pairs as a JSON-encoded map on the
// context update topic, so they can be merged into the mode context for context-aware transitions.
func (a *ApproachingPerson) updateContext(ctx map[string]any) {
	if a.session == nil {
		a.log.Warn("ApproachingPerson: no session, cannot update context")
		return
	}

	payload, err := json.Marshal(ctx)
	if err != nil {
		a.log.Error("ApproachingPerson: failed to marshal context", zap.Error(err))
		return
	}

	if err := a.session.Put(contextUpdateTopic, payload); err != nil {
		a.log.Error("ApproachingPerson: failed to publish context update", zap.Error(err))
		return
	}
	a.log.Debug("ApproachingPerson: sent context update", zap.ByteString("context", payload))
}

// Run performs one iteration of the task: it emits a SWITCH status unless a
// person has already been approached, then waits approachingPollInterval.
func (a *ApproachingPerson) Run(ctx context.Context) {
	a.mu.Lock()
	approached := a.isPersonApproached
	a.mu.Unlock()

	if approached {
		a.log.Debug("Skipping SWITCH status - person already approached.")
		util.Sleep(ctx, approachingPollInterval)
		return
	}

	if a.session == nil {
		a.log.Warn("No Zenoh session available in ApproachingPerson.")
		util.Sleep(ctx, approachingPollInterval)
		return
	}

	requestID := uuid.New().String()
	payload := serializePersonGreetingStatus(requestID, personStatusSwitch, "Switching to next person")
	if err := a.session.Put(personGreetingTopic, payload); err != nil {
		a.log.Error("ApproachingPerson: failed to publish SWITCH status", zap.Error(err))
	}

	util.Sleep(ctx, approachingPollInterval)
}

// Stop releases the zenoh subscriber and session.
func (a *ApproachingPerson) Stop() {
	a.log.Info("Stopping ApproachingPerson background task.")
	if a.sub != nil {
		a.sub.Drop()
		a.sub = nil
	}
	if a.session != nil {
		a.session.Close()
		a.session = nil
	}
}

// serializePersonGreetingStatus encodes a PersonGreetingStatus in CDR
// little-endian format, matching the Python pycdr2 wire layout. The header's
// frame_id and the request_id field both carry requestID, as in the Python task.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8) + padding to 4-byte
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] status           int8
//	[..] padding to 4-byte data boundary (before message uint32 length)
//	[..] message          length uint32 + bytes+NUL (last field, no padding)
func serializePersonGreetingStatus(requestID string, status int8, message string) []byte {
	now := time.Now()

	buf := make([]byte, 0, 200)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenohsession.AppendCDRString(buf, requestID) // frame_id, padded

	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	buf = append(buf, byte(status))

	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	msgBytes := append([]byte(message), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(msgBytes)))
	buf = append(buf, msgBytes...)

	return buf
}

// deserializePersonGreetingStatus extracts the status field from a CDR-encoded
// PersonGreetingStatus payload.
func deserializePersonGreetingStatus(data []byte) (int8, error) {
	if len(data) < 16 {
		return 0, fmt.Errorf("ApproachingPerson: payload too short")
	}

	pos := 4 // skip CDR header
	pos += 4 // stamp.sec
	pos += 4 // stamp.nanosec

	// frame_id: length uint32 + bytes + padding to 4-byte data boundary
	if pos+4 > len(data) {
		return 0, fmt.Errorf("ApproachingPerson: truncated at frame_id length")
	}
	frameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + frameIDLen
	if dataOff := pos - 4; (4-dataOff%4)%4 > 0 {
		pos += (4 - dataOff%4) % 4
	}

	// request_id: length uint32 + bytes (no padding — next field is int8)
	if pos+4 > len(data) {
		return 0, fmt.Errorf("ApproachingPerson: truncated at request_id length")
	}
	reqIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + reqIDLen

	// status: int8
	if pos >= len(data) {
		return 0, fmt.Errorf("ApproachingPerson: truncated at status")
	}
	return int8(data[pos]), nil
}
