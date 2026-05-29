package emotion

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	avatarRequestTopic = "om/avatar/request"
	switchFaceCode     = byte(0x00)
)

func init() {
	actions.Register("emotion/zenoh", newZenohConnector)
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
		log.Warn("emotion/zenoh: zenoh unavailable, avatar emotions disabled", zap.Error(err))
		return &zenohConnector{log: log}, nil
	}

	pub, err := sess.DeclarePublisher(avatarRequestTopic)
	if err != nil {
		sess.Close()
		log.Warn("emotion/zenoh: failed to declare publisher, avatar emotions disabled", zap.Error(err))
		return &zenohConnector{log: log}, nil
	}

	log.Info("emotion/zenoh: zenoh session opened")
	return &zenohConnector{log: log, session: sess, publisher: pub}, nil
}

func (z *zenohConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("emotion/zenoh: unexpected input type %T", input)
	}
	emotion, _ := args["action"].(string)
	if emotion == "" {
		return nil, nil
	}

	if z.publisher == nil {
		return nil, nil
	}

	payload := serializeAvatarRequest(emotion)
	if err := z.publisher.Put(payload); err != nil {
		z.log.Error("emotion/zenoh: put failed", zap.Error(err))
		return nil, err
	}

	z.log.Info("emotion/zenoh: published", zap.String("emotion", emotion))
	return nil, nil
}

func (z *zenohConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

func (z *zenohConnector) Stop() {
	if z.publisher != nil {
		z.publisher.Drop()
		z.publisher = nil
		z.log.Info("emotion/zenoh: publisher dropped")
	}

	if z.session != nil {
		z.session.Close()
		z.session = nil
		z.log.Info("emotion/zenoh: zenoh session closed")
	}
}

// serializeAvatarRequest encodes an AvatarFaceRequest in CDR little-endian format.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8, 4-byte aligned)
//	[..] request_id.data  CDR string (4-byte aligned after frame_id)
//	[..] code             int8 = 0 (SWITCH_FACE)
//	[..] padding to 4-byte data boundary
//	[..] face_text.data   CDR string
func serializeAvatarRequest(faceText string) []byte {
	now := time.Now()
	requestID := uuid.New().String()

	buf := make([]byte, 0, 128)

	// CDR encapsulation header (little-endian)
	buf = append(buf, 0x00, 0x01, 0x00, 0x00)

	// stamp.sec (int32 LE, data offset 0)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))

	// stamp.nanosec (uint32 LE, data offset 4)
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))

	// header.frame_id CDR string (data offset 8, 4-byte aligned)
	buf = zenohsession.AppendCDRString(buf, requestID)

	// request_id.data CDR string (4-byte aligned after frame_id padding)
	buf = zenohsession.AppendCDRString(buf, requestID)

	// code = 0 (SWITCH_FACE)
	buf = append(buf, switchFaceCode)

	// pad to 4-byte data boundary before face_text's uint32 length
	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	// face_text.data CDR string
	faceBytes := append([]byte(faceText), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(faceBytes)))
	buf = append(buf, faceBytes...)

	return buf
}
