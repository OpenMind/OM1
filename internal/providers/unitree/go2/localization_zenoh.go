package go2

import (
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/geometry"
	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	defaultLocalizationTopic = "om/localization_pose"
	defaultQualityTolerance  = 0.7
)

type localization struct {
	pose           geometry.Pose
	matchScore     int32
	qualityPercent float32
	numPoints      int32
}

type LocalizationProvider struct {
	log              *zap.Logger
	topic            string
	qualityTolerance float32
	session          zenohsession.Session
	sub              zenohsession.Subscriber

	localized atomic.Bool
	pose      atomic.Pointer[geometry.Pose]

	lastDebugLog time.Time
}

var (
	localizationOnce     sync.Once
	localizationInstance *LocalizationProvider
)

func Localization() *LocalizationProvider {
	localizationOnce.Do(func() {
		localizationInstance = NewLocalizationProvider("", 0)
	})
	return localizationInstance
}

// NewLocalizationProvider opens a zenoh session and subscribes to topic.
func NewLocalizationProvider(topic string, qualityTolerance float32) *LocalizationProvider {
	if topic == "" {
		topic = defaultLocalizationTopic
	}
	if qualityTolerance <= 0 {
		qualityTolerance = defaultQualityTolerance
	}

	p := &LocalizationProvider{
		log:              logger.Get().Named("unitree_go2_localization"),
		topic:            topic,
		qualityTolerance: qualityTolerance,
	}

	sess, err := zenohsession.Open()
	if err != nil {
		p.log.Warn("zenoh unavailable, provider disabled", zap.Error(err))
		return p
	}
	p.session = sess

	sub, err := sess.DeclareSubscriber(topic, p.onSample)
	if err != nil {
		sess.Close()
		p.session = nil
		p.log.Warn("failed to declare subscriber", zap.Error(err))
		return p
	}
	p.sub = sub

	p.log.Info("provider initialized",
		zap.String("topic", topic),
		zap.Float32("quality_tolerance", qualityTolerance),
	)
	return p
}

// onSample decodes an incoming Localization message and updates state.
func (p *LocalizationProvider) onSample(data []byte) {
	if len(data) == 0 {
		p.log.Warn("received empty message")
		return
	}

	msg, err := deserializeLocalization(data)
	if err != nil {
		p.log.Error("failed to decode message", zap.Error(err))
		return
	}
	p.process(msg)
}

// process publishes the decoded localization state to the lock-free readers.
func (p *LocalizationProvider) process(msg localization) {
	localized := msg.qualityPercent >= p.qualityTolerance

	p.localized.Store(localized)
	p.pose.Store(&msg.pose)

	if now := time.Now(); now.Sub(p.lastDebugLog) >= debugLogInterval {
		p.lastDebugLog = now
		p.log.Debug("go2 localization",
			zap.Bool("localized", localized),
			zap.Float32("quality_percent", msg.qualityPercent),
			zap.Float64("x", msg.pose.Position.X),
			zap.Float64("y", msg.pose.Position.Y),
			zap.Float64("z", msg.pose.Position.Z),
		)
	}
}

// IsLocalized reports whether the latest fix meets the quality tolerance.
func (p *LocalizationProvider) IsLocalized() bool {
	return p.localized.Load()
}

// Pose returns the latest localization pose, or nil if none has been received.
func (p *LocalizationProvider) Pose() *geometry.Pose {
	return p.pose.Load()
}

// Stop releases the zenoh subscriber and session.
func (p *LocalizationProvider) Stop() {
	if p.sub != nil {
		p.sub.Drop()
		p.sub = nil
	}

	if p.session != nil {
		p.session.Close()
		p.session = nil
	}

	p.log.Info("provider stopped")
}

// deserializeLocalization decodes a CDR-encoded Localization message.
//
// Wire layout (offsets relative to the start of the buffer):
//
//	[0]  CDR encapsulation header: 4 bytes
//	[4]  header.stamp.sec       int32  LE  (data offset 0)
//	[8]  header.stamp.nanosec   int32  LE  (data offset 4)
//	[12] header.frame_id        CDR string (data offset 8) + padding to 4-byte
//	[..] padding to 8-byte data boundary (float64 alignment)
//	[..] pose.position.x/y/z          float64 LE x3
//	[..] pose.orientation.x/y/z/w     float64 LE x4
//	[..] match_score      int32   LE
//	[..] quality_percent  float32 LE
//	[..] num_points       int32   LE
func deserializeLocalization(data []byte) (localization, error) {
	var m localization

	if len(data) < 16 {
		return m, fmt.Errorf("payload too short (%d bytes)", len(data))
	}

	// Skip CDR encapsulation header (4 bytes) and header.stamp (sec + nanosec).
	pos := 4 + 8

	if pos+4 > len(data) {
		return m, fmt.Errorf("truncated at frame_id length")
	}
	frameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + frameIDLen
	if dataOff := pos - 4; (4-dataOff%4)%4 > 0 {
		pos += (4 - dataOff%4) % 4
	}

	// Align to 8-byte boundary for the pose float64s.
	if dataOff := pos - 4; (8-dataOff%8)%8 > 0 {
		pos += (8 - dataOff%8) % 8
	}

	const nFloats = 7
	if pos+nFloats*8+12 > len(data) {
		return m, fmt.Errorf("truncated at pose/trailer")
	}

	readF64 := func() float64 {
		v := zenohsession.ReadFloat64LE(data, pos)
		pos += 8
		return v
	}

	m.pose.Position.X = readF64()
	m.pose.Position.Y = readF64()
	m.pose.Position.Z = readF64()
	m.pose.Orientation.X = readF64()
	m.pose.Orientation.Y = readF64()
	m.pose.Orientation.Z = readF64()
	m.pose.Orientation.W = readF64()

	m.matchScore = int32(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	m.qualityPercent = zenohsession.ReadFloat32LE(data, pos)
	pos += 4
	m.numPoints = int32(binary.LittleEndian.Uint32(data[pos:]))

	return m, nil
}
