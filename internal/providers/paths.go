package providers

import (
	"encoding/binary"
	"fmt"
	"strings"
	"sync"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const pathsTopic = "om/paths"

// Movement encapsulates the derived movement options based on the current valid paths.
type Movement struct {
	TurnLeft  []uint32
	TurnRight []uint32
	Advance   []uint32
	Retreat   bool
}

// PathsProvider subscribes to the om/paths topic and derives a natural-language
// summary of the safe movement directions from the raw path indices.
//
// Path indices are classified into movement options as follows:
//
//	[0, 3)  -> turn left
//	[3, 5]  -> move forwards
//	(5, 9)  -> turn right
//	9       -> move back (retreat)
type PathsProvider struct {
	log     *zap.Logger
	session *zenohsession.Session
	sub     *zenohsession.Subscriber

	mu          sync.RWMutex
	turnLeft    []uint32
	turnRight   []uint32
	advance     []uint32
	retreat     bool
	validPaths  []uint32
	lidarString string
}

// NewPathsProvider opens a zenoh session and subscribes to the om/paths topic.
func NewPathsProvider() *PathsProvider {
	p := &PathsProvider{log: logger.Get()}

	sess, err := zenohsession.Open()
	if err != nil {
		p.log.Warn("paths: zenoh unavailable, paths disabled", zap.Error(err))
		return p
	}
	p.session = sess

	sub, err := sess.DeclareSubscriber(pathsTopic, p.onPaths)
	if err != nil {
		sess.Close()
		p.session = nil
		p.log.Warn("paths: failed to declare subscriber", zap.Error(err))
		return p
	}
	p.sub = sub

	p.log.Info("paths: provider initialized", zap.String("topic", pathsTopic))
	return p
}

// onPaths handles an incoming Paths message, recomputing the derived movement
// options and the natural-language summary.
func (p *PathsProvider) onPaths(data []byte) {
	paths, err := deserializePaths(data)
	if err != nil {
		p.log.Error("paths: failed to deserialize message", zap.Error(err))
		return
	}

	var turnLeft, turnRight, advance []uint32
	retreat := false

	for _, path := range paths {
		switch {
		case path < 3:
			turnLeft = append(turnLeft, path)
		case path <= 5:
			advance = append(advance, path)
		case path < 9:
			turnRight = append(turnRight, path)
		case path == 9:
			retreat = true
		}
	}

	p.mu.Lock()
	p.turnLeft = turnLeft
	p.turnRight = turnRight
	p.advance = advance
	p.retreat = retreat
	p.validPaths = paths
	p.lidarString = generateMovementString(paths, turnLeft, advance, turnRight, retreat)
	p.mu.Unlock()
}

// generateMovementString builds the natural-language assessment of possible
// movement directions from the derived movement options.
func generateMovementString(validPaths, turnLeft, advance, turnRight []uint32, retreat bool) string {
	if validPaths == nil {
		return "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
	}

	var b strings.Builder
	b.WriteString("The safe movement directions are: {")

	if len(turnLeft) > 0 {
		b.WriteString("'turn left', ")
	}
	if len(advance) > 0 {
		b.WriteString("'move forwards', ")
	}
	if len(turnRight) > 0 {
		b.WriteString("'turn right', ")
	}
	if retreat {
		b.WriteString("'move back', ")
	}

	b.WriteString("'stand still'}. ")
	return b.String()
}

// LidarString returns the latest natural-language assessment of possible paths,
// or an empty string if no message has been received yet.
func (p *PathsProvider) LidarString() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lidarString
}

// ValidPaths returns the most recently received path indices.
func (p *PathsProvider) ValidPaths() []uint32 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.validPaths
}

// MovementOptions returns the derived movement options based on the current
// valid paths.
func (p *PathsProvider) MovementOptions() map[string]any {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return map[string]any{
		"turn_left":  p.turnLeft,
		"advance":    p.advance,
		"turn_right": p.turnRight,
		"retreat":    p.retreat,
	}
}

// Movement returns the derived movement options as a Movement struct.
func (p *PathsProvider) Movement() Movement {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return Movement{
		TurnLeft:  append([]uint32(nil), p.turnLeft...),
		TurnRight: append([]uint32(nil), p.turnRight...),
		Advance:   append([]uint32(nil), p.advance...),
		Retreat:   p.retreat,
	}
}

// Stop releases the zenoh subscriber and session.
func (p *PathsProvider) Stop() {
	if p.sub != nil {
		p.sub.Drop()
		p.sub = nil
	}
	if p.session != nil {
		p.session.Close()
		p.session = nil
	}
	p.log.Info("paths: provider stopped")
}

// deserializePaths extracts the path indices from a CDR-encoded Paths message.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 4 bytes
//	[4]  stamp.sec        int32  LE  (data offset 0)
//	[8]  stamp.nanosec    uint32 LE  (data offset 4)
//	[12] header.frame_id  CDR string (data offset 8) + padding to 4-byte
//	[..] paths            sequence<uint32>: count uint32 + count*uint32
//	[..] blocked_by_obstacle_idx, blocked_by_hazard_idx (ignored)
func deserializePaths(data []byte) ([]uint32, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("paths: payload too short (%d bytes)", len(data))
	}

	pos := 4 // skip CDR header
	pos += 4 // stamp.sec
	pos += 4 // stamp.nanosec

	// frame_id: length uint32 + bytes + padding to 4-byte data boundary
	if pos+4 > len(data) {
		return nil, fmt.Errorf("paths: truncated at frame_id length")
	}
	frameIDLen := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4 + frameIDLen
	if dataOff := pos - 4; (4-dataOff%4)%4 > 0 {
		pos += (4 - dataOff%4) % 4
	}

	// paths: sequence<uint32> — count uint32 followed by count uint32 values
	if pos+4 > len(data) {
		return nil, fmt.Errorf("paths: truncated at paths count")
	}
	count := int(binary.LittleEndian.Uint32(data[pos:]))
	pos += 4
	if pos+count*4 > len(data) {
		return nil, fmt.Errorf("paths: truncated at paths data (count=%d)", count)
	}

	paths := make([]uint32, count)
	for i := 0; i < count; i++ {
		paths[i] = binary.LittleEndian.Uint32(data[pos:])
		pos += 4
	}

	return paths, nil
}
