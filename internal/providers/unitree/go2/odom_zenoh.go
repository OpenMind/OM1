package go2

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	defaultOdomTopic = "odom"
	radToDeg         = 57.2958
)

type RobotState string

const (
	RobotStateStanding RobotState = "standing"
	RobotStateSitting  RobotState = "sitting"
)

type OdomPosition struct {
	OdomX            float64    `json:"odom_x"`
	OdomY            float64    `json:"odom_y"`
	Moving           bool       `json:"moving"`
	OdomYaw0360      float64    `json:"odom_yaw_0_360"`
	OdomYawM180P180  float64    `json:"odom_yaw_m180_p180"`
	BodyHeightCm     int        `json:"body_height_cm"`
	BodyAttitude     RobotState `json:"body_attitude"`
	OdomRockchipTS   float64    `json:"odom_rockchip_ts"`
	OdomSubscriberTS float64    `json:"odom_subscriber_ts"`
}

type OdomZenohProvider struct {
	log     *zap.Logger
	topic   string
	session zenohsession.Session
	sub     zenohsession.Subscriber

	mu               sync.RWMutex
	bodyHeightCm     int
	bodyAttitude     RobotState
	moving           bool
	previousX        float64
	previousY        float64
	previousZ        float64
	moveHistory      float64
	x                float64
	y                float64
	z                float64
	odomYaw0360      float64
	odomYawM180P180  float64
	odomRockchipTS   float64
	odomSubscriberTS float64

	lastDebugLog time.Time
}

var (
	odomOnce     sync.Once
	odomInstance *OdomZenohProvider
)

// OdomZenoh returns the singleton provider subscribed to the default topic.
func OdomZenoh() *OdomZenohProvider {
	odomOnce.Do(func() { odomInstance = NewOdomZenohProvider("") })
	return odomInstance
}

// NewOdomZenohProvider opens a zenoh session and subscribes to topic.
func NewOdomZenohProvider(topic string) *OdomZenohProvider {
	if topic == "" {
		topic = defaultOdomTopic
	}

	p := &OdomZenohProvider{log: logger.Get(), topic: topic}

	sess, err := zenohsession.Open()
	if err != nil {
		p.log.Warn("go2 odom: zenoh unavailable, provider disabled", zap.Error(err))
		return p
	}
	p.session = sess

	sub, err := sess.DeclareSubscriber(topic, p.onSample)
	if err != nil {
		sess.Close()
		p.session = nil
		p.log.Warn("go2 odom: failed to declare subscriber", zap.Error(err))
		return p
	}
	p.sub = sub

	p.log.Info("go2 odom: provider initialized", zap.String("topic", topic))
	return p
}

// onSample decodes an incoming PoseStamped message and updates odometry state.
func (p *OdomZenohProvider) onSample(data []byte) {
	pose, err := deserializePoseStamped(data)
	if err != nil {
		p.log.Error("go2 odom: failed to decode PoseStamped", zap.Error(err))
		return
	}
	p.processOdom(pose)
}

type poseStamped struct {
	stampSec     int32
	stampNanosec uint32
	posX         float64
	posY         float64
	posZ         float64
	oriX         float64
	oriY         float64
	oriZ         float64
	oriW         float64
}

// processOdom updates the internal odometry state from a decoded pose.
func (p *OdomZenohProvider) processOdom(pose poseStamped) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.odomRockchipTS = float64(pose.stampSec) + float64(pose.stampNanosec)*1e-9
	p.odomSubscriberTS = float64(time.Now().UnixNano()) / 1e9

	p.bodyHeightCm = int(math.Round(pose.posZ * 100.0))
	switch {
	case p.bodyHeightCm > 24:
		p.bodyAttitude = RobotStateStanding
	case p.bodyHeightCm > 3:
		p.bodyAttitude = RobotStateSitting
	}

	dx := (pose.posX - p.previousX) * (pose.posX - p.previousX)
	dy := (pose.posY - p.previousY) * (pose.posY - p.previousY)
	dz := (pose.posZ - p.previousZ) * (pose.posZ - p.previousZ)

	p.previousX = pose.posX
	p.previousY = pose.posY
	p.previousZ = pose.posZ

	delta := math.Sqrt(dx + dy + dz)

	p.moveHistory = 0.7*delta + 0.3*p.moveHistory
	if delta > 0.01 || p.moveHistory > 0.01 {
		p.moving = true
	} else {
		p.moving = false
	}

	_, _, yaw := eulerFromQuaternion(pose.oriX, pose.oriY, pose.oriZ, pose.oriW)

	p.odomYawM180P180 = yaw * radToDeg

	flip := -1.0 * p.odomYawM180P180
	if flip < 0.0 {
		flip += 360.0
	}
	p.odomYaw0360 = flip

	p.x = pose.posX
	p.y = pose.posY

	if now := time.Now(); now.Sub(p.lastDebugLog) >= debugLogInterval {
		p.lastDebugLog = now
		p.log.Debug("go2 odom",
			zap.Float64("x", p.x), zap.Float64("y", p.y), zap.Float64("z", p.z),
			zap.String("body_attitude", string(p.bodyAttitude)),
			zap.Float64("yaw_m180_p180", p.odomYawM180P180),
			zap.Float64("yaw_0_360", p.odomYaw0360),
			zap.Float64("rockchip_ts", p.odomRockchipTS),
		)
	}
}

// Position returns the latest odometry snapshot in the world frame.
func (p *OdomZenohProvider) Position() OdomPosition {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return OdomPosition{
		OdomX:            p.x,
		OdomY:            p.y,
		Moving:           p.moving,
		OdomYaw0360:      p.odomYaw0360,
		OdomYawM180P180:  p.odomYawM180P180,
		BodyHeightCm:     p.bodyHeightCm,
		BodyAttitude:     p.bodyAttitude,
		OdomRockchipTS:   p.odomRockchipTS,
		OdomSubscriberTS: p.odomSubscriberTS,
	}
}

// Stop releases the zenoh subscriber and session.
func (p *OdomZenohProvider) Stop() {
	if p.sub != nil {
		p.sub.Drop()
		p.sub = nil
	}

	if p.session != nil {
		p.session.Close()
		p.session = nil
	}

	p.log.Info("go2 odom: provider stopped")
}

// eulerFromQuaternion converts a quaternion to roll, pitch, yaw in radians.
func eulerFromQuaternion(x, y, z, w float64) (roll, pitch, yaw float64) {
	t0 := +2.0 * (w*x + y*z)
	t1 := +1.0 - 2.0*(x*x+y*y)
	roll = math.Atan2(t0, t1)

	t2 := +2.0 * (w*y - z*x)
	if t2 > +1.0 {
		t2 = +1.0
	}

	if t2 < -1.0 {
		t2 = -1.0
	}

	pitch = math.Asin(t2)

	t3 := +2.0 * (w*z + x*y)
	t4 := +1.0 - 2.0*(y*y+z*z)
	yaw = math.Atan2(t3, t4)

	return roll, pitch, yaw
}

// deserializePoseStamped decodes a CDR-encoded nav_msgs/Odometry message.
//
// Wire layout (offsets relative to the start of the buffer):
//
//	[0]  CDR encapsulation header: 4 bytes
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8) + padding to 4-byte
//	[..] child_frame_id   CDR string + padding to 4-byte
//	[..] padding to 8-byte data boundary (float64 alignment)
//	[..] pose.pose.position.x/y/z        float64 LE x3
//	[..] pose.pose.orientation.x/y/z/w   float64 LE x4
//	[..] pose.covariance, twist, twist.covariance (unused)
func deserializePoseStamped(data []byte) (poseStamped, error) {
	var ps poseStamped

	if len(data) < 16 {
		return ps, fmt.Errorf("go2 odom: payload too short (%d bytes)", len(data))
	}

	pos := 4
	ps.stampSec = int32(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	ps.stampNanosec = binary.LittleEndian.Uint32(data[pos:])
	pos += 4

	if pos+4 > len(data) {
		return ps, fmt.Errorf("go2 odom: truncated at frame_id length")
	}

	frameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + frameIDLen
	if dataOff := pos - 4; (4-dataOff%4)%4 > 0 {
		pos += (4 - dataOff%4) % 4
	}

	// child_frame_id
	if pos+4 > len(data) {
		return ps, fmt.Errorf("go2 odom: truncated at child_frame_id length")
	}
	childFrameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + childFrameIDLen

	if dataOff := pos - 4; (8-dataOff%8)%8 > 0 {
		pos += (8 - dataOff%8) % 8
	}

	const nFloats = 7
	if pos+nFloats*8 > len(data) {
		return ps, fmt.Errorf("go2 odom: truncated at pose floats")
	}

	readF64 := func() float64 {
		v := zenohsession.ReadFloat64LE(data, pos)
		pos += 8
		return v
	}

	ps.posX = readF64()
	ps.posY = readF64()
	ps.posZ = readF64()
	ps.oriX = readF64()
	ps.oriY = readF64()
	ps.oriZ = readF64()
	ps.oriW = readF64()

	return ps, nil
}
