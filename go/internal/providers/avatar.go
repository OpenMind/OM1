package providers

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	avatarRequestTopic  = "om/avatar/request"
	avatarResponseTopic = "om/avatar/response"

	avatarCodeSwitchFace = byte(0x00) // AvatarFaceRequest.Code.SWITCH_FACE
	avatarCodeStatus     = byte(0x01) // AvatarFaceRequest.Code.STATUS
	avatarCodeActive     = byte(0x00) // AvatarFaceResponse.Code.ACTIVE
)

// AvatarProvider is a singleton that manages the zenoh session and publishers/subscriber
// for avatar communication.
type AvatarProvider struct {
	log               *zap.Logger
	session           *zenohsession.Session
	requestPublisher  *zenohsession.Publisher  // publishes to om/avatar/request (face commands)
	responsePublisher *zenohsession.Publisher  // publishes to om/avatar/response (health check replies)
	requestSubscriber *zenohsession.Subscriber // subscribes to om/avatar/request
}

var (
	avatarOnce     sync.Once
	avatarInstance *AvatarProvider
)

// Avatar returns the singleton AvatarProvider.
func Avatar(endpoint string) *AvatarProvider {
	avatarOnce.Do(func() { avatarInstance = NewAvatarProvider(endpoint) })
	return avatarInstance
}

// NewAvatarProvider initializes the AvatarProvider by setting up a zenoh session, publishers, and subscriber.
func NewAvatarProvider(endpoint string) *AvatarProvider {
	p := &AvatarProvider{log: logger.Get()}

	sess, err := zenohsession.Open(endpoint)
	if err != nil {
		p.log.Warn("avatar: zenoh unavailable, avatar disabled", zap.Error(err))
		return p
	}
	p.session = sess

	requestPub, err := sess.DeclarePublisher(avatarRequestTopic)
	if err != nil {
		sess.Close()
		p.session = nil
		p.log.Warn("avatar: failed to declare request publisher", zap.Error(err))
		return p
	}
	p.requestPublisher = requestPub

	responsePub, err := sess.DeclarePublisher(avatarResponseTopic)
	if err != nil {
		requestPub.Drop()
		sess.Close()
		p.requestPublisher = nil
		p.session = nil
		p.log.Warn("avatar: failed to declare response publisher", zap.Error(err))
		return p
	}
	p.responsePublisher = responsePub

	sub, err := sess.DeclareSubscriber(avatarRequestTopic, p.handleRequest)
	if err != nil {
		responsePub.Drop()
		requestPub.Drop()
		sess.Close()
		p.responsePublisher = nil
		p.requestPublisher = nil
		p.session = nil
		p.log.Warn("avatar: failed to declare subscriber", zap.Error(err))
		return p
	}
	p.requestSubscriber = sub

	p.log.Info("avatar: provider initialized",
		zap.String("request_topic", avatarRequestTopic),
		zap.String("response_topic", avatarResponseTopic),
	)
	return p
}

// handleRequest processes incoming avatar requests.
func (p *AvatarProvider) handleRequest(data []byte) {
	code, requestID, _, err := deserializeAvatarRequest(data)
	if err != nil {
		p.log.Error("avatar: failed to deserialize request", zap.Error(err))
		return
	}

	if code != avatarCodeStatus {
		return
	}

	if p.responsePublisher == nil {
		return
	}

	payload := serializeAvatarResponse(requestID, avatarCodeActive, "Avatar system active")
	if err := p.responsePublisher.Put(payload); err != nil {
		p.log.Error("avatar: failed to publish health check response", zap.Error(err))
		return
	}
}

// SendAvatarCommand publishes a SWITCH_FACE request for the given face expression.
func (p *AvatarProvider) SendAvatarCommand(command string) error {
	if p.requestPublisher == nil {
		return fmt.Errorf("avatar: publisher not available")
	}

	payload := serializeAvatarRequest(command)
	if err := p.requestPublisher.Put(payload); err != nil {
		return fmt.Errorf("avatar: put failed: %w", err)
	}

	p.log.Info("avatar: published command", zap.String("command", command))
	return nil
}

// serializeAvatarRequest encodes an AvatarFaceRequest (SWITCH_FACE) in CDR
// little-endian format, matching the Python pycdr2 wire layout.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8) + padding to 4-byte
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8 = 0x00 (SWITCH_FACE)
//	[..] padding to 4-byte data boundary (before face_text uint32 length)
//	[..] face_text        length uint32 + bytes+NUL (last field, no padding)
func serializeAvatarRequest(faceText string) []byte {
	now := time.Now()
	requestID := uuid.New().String()

	buf := make([]byte, 0, 200)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenohsession.AppendCDRString(buf, requestID) // frame_id, padded

	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	buf = append(buf, avatarCodeSwitchFace)

	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	faceBytes := append([]byte(faceText), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(faceBytes)))
	buf = append(buf, faceBytes...)

	return buf
}

// serializeAvatarResponse encodes an AvatarFaceResponse in CDR little-endian
// format, matching the Python pycdr2 wire layout.
//
// Wire layout:
//
//	[0]  CDR header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
//	[..] padding to 4-byte data boundary (before message uint32 length)
//	[..] message          length uint32 + bytes+NUL (last field, no padding)
func serializeAvatarResponse(requestID string, code byte, message string) []byte {
	now := time.Now()
	responseID := uuid.New().String()

	buf := make([]byte, 0, 200)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenohsession.AppendCDRString(buf, responseID) // frame_id, padded

	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	buf = append(buf, code)

	dataLen := len(buf) - 4
	if pad := (4 - dataLen%4) % 4; pad > 0 {
		buf = append(buf, make([]byte, pad)...)
	}

	msgBytes := append([]byte(message), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(msgBytes)))
	buf = append(buf, msgBytes...)

	return buf
}

// deserializeAvatarRequest decodes the code, request_id, and face_text fields
// from a CDR-encoded AvatarFaceRequest payload.
func deserializeAvatarRequest(data []byte) (code byte, requestID, faceText string, err error) {
	if len(data) < 16 {
		return 0, "", "", fmt.Errorf("avatar: payload too short (%d bytes)", len(data))
	}

	pos := 4 // skip CDR header
	pos += 4 // stamp.sec
	pos += 4 // stamp.nanosec

	// frame_id: AppendCDRString format — length uint32 + bytes + padding to 4-byte data boundary
	if pos+4 > len(data) {
		return 0, "", "", fmt.Errorf("avatar: truncated at frame_id length")
	}
	frameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + frameIDLen
	dataOff := pos - 4
	if pad := (4 - dataOff%4) % 4; pad > 0 {
		pos += pad
	}

	// request_id: length uint32 + bytes (no padding — next field is int8)
	if pos+4 > len(data) {
		return 0, "", "", fmt.Errorf("avatar: truncated at request_id length")
	}
	reqIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	if pos+reqIDLen > len(data) {
		return 0, "", "", fmt.Errorf("avatar: truncated at request_id data")
	}
	reqIDBytes := data[pos : pos+reqIDLen]
	if reqIDLen > 0 && reqIDBytes[reqIDLen-1] == 0 {
		reqIDBytes = reqIDBytes[:reqIDLen-1]
	}
	requestID = string(reqIDBytes)
	pos += reqIDLen

	// code: int8
	if pos >= len(data) {
		return 0, "", "", fmt.Errorf("avatar: truncated at code")
	}
	code = data[pos]
	pos++

	// padding to 4-byte data boundary (before face_text length uint32)
	dataOff = pos - 4
	if pad := (4 - dataOff%4) % 4; pad > 0 {
		pos += pad
	}

	// face_text: length uint32 + bytes
	if pos+4 > len(data) {
		return code, requestID, "", nil // face_text absent (e.g. STATUS request)
	}
	faceLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	if pos+faceLen > len(data) {
		return code, requestID, "", fmt.Errorf("avatar: truncated at face_text data")
	}
	faceBytes := data[pos : pos+faceLen]
	if faceLen > 0 && faceBytes[faceLen-1] == 0 {
		faceBytes = faceBytes[:faceLen-1]
	}
	faceText = string(faceBytes)

	return code, requestID, faceText, nil
}
