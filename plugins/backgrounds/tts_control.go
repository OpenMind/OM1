package backgrounds

import (
	"context"
	"encoding/binary"
	"fmt"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	ttsRequestTopic  = "om/tts/request"
	ttsResponseTopic = "om/tts/response"

	// OMTTSRequest.code values.
	ttsCodeDisable byte = 0
	ttsCodeEnable  byte = 1
	ttsCodeStatus  byte = 2
)

func init() {
	bg.Register("TTSControl", NewTTSControl)
}

// TTSControl mutes or unmutes the robot's speech from OMTTSRequest messages on
// om/tts/request, and acks on om/tts/response.
type TTSControl struct {
	log     *zap.Logger
	session zenohsession.Session
	sub     zenohsession.Subscriber
}

// NewTTSControl subscribes to the TTS-request topic.
func NewTTSControl(_ map[string]any) (bg.Background, error) {
	log := logger.Get().Named("TTSControl")

	t := &TTSControl{log: log}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Error("zenoh unavailable, TTS control disabled", zap.Error(err))
		return t, nil
	}
	t.session = sess

	sub, err := sess.DeclareSubscriber(ttsRequestTopic, t.onTTSRequest)
	if err != nil {
		log.Error("failed to declare subscriber", zap.Error(err))
	} else {
		t.sub = sub
	}

	log.Info("background task initialized")
	return t, nil
}

// onTTSRequest decodes an OMTTSRequest, applies the suppression state, and acks.
func (t *TTSControl) onTTSRequest(data []byte) {
	req, err := deserializeTTSRequest(data)
	if err != nil {
		t.log.Error("failed to deserialize TTS request", zap.Error(err))
		return
	}

	switch req.code {
	case ttsCodeDisable:
		tts.SetSuppressed(true)
		t.log.Info("TTS disabled")
	case ttsCodeEnable:
		tts.SetSuppressed(false)
		t.log.Info("TTS enabled")
	case ttsCodeStatus:
		// Status request: report current state without changing it.
	default:
		t.log.Warn("unknown TTS request code", zap.Uint8("code", req.code))
		return
	}

	t.publishResponse(req.frameID, req.requestID)
}

// publishResponse acks the current suppression state on om/tts/response.
func (t *TTSControl) publishResponse(frameID, requestID string) {
	if t.session == nil {
		return
	}

	code := ttsCodeEnable
	status := "TTS Enabled"
	if tts.Suppressed.Load() {
		code = ttsCodeDisable
		status = "TTS Disabled"
	}

	payload := serializeTTSResponse(frameID, requestID, code, status)
	if err := t.session.Put(ttsResponseTopic, payload); err != nil {
		t.log.Error("failed to publish TTS response", zap.Error(err))
	}
}

// Run blocks until shutdown; work is driven by the subscriber callback.
func (t *TTSControl) Run(ctx context.Context) {
	<-ctx.Done()
}

// Stop releases the zenoh subscriber and session.
func (t *TTSControl) Stop() {
	t.log.Info("stopping background task")
	if t.sub != nil {
		t.sub.Drop()
		t.sub = nil
	}
	if t.session != nil {
		t.session.Close()
		t.session = nil
	}
}

// ttsRequest is the decoded payload of an OMTTSRequest message.
type ttsRequest struct {
	frameID   string
	requestID string
	code      byte
}

// deserializeTTSRequest decodes the header frame_id, request_id, and code
// fields from a CDR-encoded OMTTSRequest payload.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 4 bytes
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte data boundary
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
func deserializeTTSRequest(data []byte) (ttsRequest, error) {
	var req ttsRequest

	if len(data) < 16 {
		return req, fmt.Errorf("tts request: payload too short (%d bytes)", len(data))
	}

	pos := 4 // skip CDR header
	pos += 4 // stamp.sec
	pos += 4 // stamp.nanosec

	frameID, next, err := readTTSCDRString(data, pos, true)
	if err != nil {
		return req, fmt.Errorf("tts request: frame_id: %w", err)
	}
	req.frameID = frameID
	pos = next

	requestID, next, err := readTTSCDRString(data, pos, false)
	if err != nil {
		return req, fmt.Errorf("tts request: request_id: %w", err)
	}
	req.requestID = requestID
	pos = next

	if pos >= len(data) {
		return req, fmt.Errorf("tts request: truncated at code")
	}
	req.code = data[pos]

	return req, nil
}

func readTTSCDRString(data []byte, off int, padded bool) (string, int, error) {
	if off+4 > len(data) {
		return "", off, fmt.Errorf("truncated at length")
	}
	strLen := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if off+strLen > len(data) {
		return "", off, fmt.Errorf("truncated at data (len=%d)", strLen)
	}

	b := data[off : off+strLen]
	if strLen > 0 && b[strLen-1] == 0 {
		b = b[:strLen-1]
	}
	off += strLen

	if padded {
		dataOff := off - 4
		off += (4 - dataOff%4) % 4
	}

	return string(b), off, nil
}

// serializeTTSResponse encodes an OMTTSResponse message in CDR little-endian
// format.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  stamp.sec        int32  LE
//	[8]  stamp.nanosec    uint32 LE
//	[12] header.frame_id  CDR string + padding to 4-byte data boundary
//	[..] request_id       length uint32 + bytes+NUL — NO padding (next is int8)
//	[..] code             int8
//	[..] padding to 4-byte data boundary (before status uint32 length)
//	[..] status           length uint32 + bytes+NUL (last field, no padding)
func serializeTTSResponse(frameID, requestID string, code byte, status string) []byte {
	now := time.Now()

	buf := make([]byte, 0, 128)

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendInt32LE(buf, int32(now.Unix()))
	buf = zenohsession.AppendUint32LE(buf, uint32(now.Nanosecond()))
	buf = zenohsession.AppendCDRString(buf, frameID) // header.frame_id, padded

	// request_id String: length uint32 + bytes (no padding — next field is int8)
	reqBytes := append([]byte(requestID), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(reqBytes)))
	buf = append(buf, reqBytes...)

	// code: int8
	buf = append(buf, code)

	// padding to 4-byte data boundary (before status length uint32)
	dataLen := len(buf) - 4 // padding is measured from the start of CDR data, not the buffer
	buf = append(buf, make([]byte, (4-dataLen%4)%4)...)

	// status String: length uint32 + bytes (last field, no padding)
	statusBytes := append([]byte(status), 0x00)
	buf = zenohsession.AppendUint32LE(buf, uint32(len(statusBytes)))
	buf = append(buf, statusBytes...)

	return buf
}
