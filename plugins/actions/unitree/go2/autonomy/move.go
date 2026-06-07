package autonomy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/unitree/go2"
	"github.com/openmind/om1/internal/util"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	defaultCmdVelTopic    = "cmd_vel"
	defaultAIRequestTopic = "om/ai/request"
	defaultAIRespTopic    = "om/ai/response"

	moveSpeed            = 0.5  // forward/back linear speed (m/s)
	turnSpeed            = 0.8  // angular speed while turning (rad/s)
	angleToleranceDeg    = 5.0  // yaw within this gap is considered "facing" the goal
	distanceTolerance    = 0.05 // metres within this gap is considered "arrived"
	movementAttemptLimit = 15   // abort a command that fails to converge in this many ticks
	advanceClearPath     = 4    // straight-ahead path index that must be clear to advance
	tickInterval         = 100 * time.Millisecond
	guardPollInterval    = 500 * time.Millisecond
)

// pathAngles maps each path index to its heading offset in degrees, matching the
// SimplePathsProvider fan-out (-60° on the left through +60° on the right, with
// index 9 reserved for retreat).
var pathAngles = [...]float64{-60, -45, -30, -15, 0, 15, 30, 45, 60, 180}

type MoveAction string

func (MoveAction) EnumValues() []string {
	return []string{
		"turn left",
		"turn right",
		"move forwards",
		"move back",
		"stand still",
	}
}

type MoveInput struct {
	Action MoveAction `json:"action" description:"The movement to perform"`
}

func init() {
	actions.RegisterInterface(
		"unitree_go2_autonomy",
		"Action interface for autonomous Unitree Go2 movement. "+
			"Validates the requested direction against lidar-derived safe paths and "+
			"drives the robot via geometry_msgs/Twist commands on the /cmd_vel Zenoh topic. "+
			"Supported movements: turn left, turn right, move forwards, move back, stand still.",
		MoveInput{},
	)
	actions.Register("unitree_go2_autonomy/move", NewMoveConnector)
}

// moveCommand is a single turn-then-advance plan derived from an AI command.
type moveCommand struct {
	dx           float64 // signed advance distance (m); positive forwards, negative back
	yaw          float64 // target heading (deg)
	startX       float64 // odometry x at command creation
	startY       float64 // odometry y at command creation
	turnComplete bool    // whether phase 1 (turning) is done
	speed        float64 // advance speed (m/s)
}

// moveConnector executes autonomous movement commands for the Unitree Go2.
type moveConnector struct {
	log *zap.Logger

	odom  *go2.OdomZenohProvider
	paths *providers.PathsProvider

	cmdVel  *zenohsession.Publisher
	session *zenohsession.Session

	aiRespPub *zenohsession.Publisher
	aiReqSub  *zenohsession.Subscriber

	aiControlEnabled atomic.Bool

	mode  string
	guard *guardWatcher

	rng *rand.Rand

	mu               sync.Mutex
	pending          *moveCommand
	movementAttempts int
	gapPrevious      float64
}

// NewMoveConnector builds the autonomy connector from its decoded config.
func NewMoveConnector(cfg map[string]any) (actions.Connector, error) {
	log := logger.Get().Named("unitree_go2_autonomy/move")

	cmdVelTopic := util.StringFrom(cfg["cmd_vel_topic"], defaultCmdVelTopic)
	aiReqTopic := util.StringFrom(cfg["ai_request_topic"], defaultAIRequestTopic)
	aiRespTopic := util.StringFrom(cfg["ai_response_topic"], defaultAIRespTopic)

	c := &moveConnector{
		log:   log,
		odom:  go2.OdomZenoh(),
		paths: providers.NewPathsProvider(),
		mode:  util.StringFrom(cfg["mode"], ""),
		rng:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	c.aiControlEnabled.Store(true)

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn("zenoh unavailable, movement disabled", zap.Error(err))
		return c, nil
	}
	c.session = sess

	if c.cmdVel, err = sess.DeclarePublisher(cmdVelTopic); err != nil {
		log.Warn("failed to declare /cmd_vel publisher, movement disabled", zap.Error(err))
		c.cmdVel = nil
	}

	if c.aiRespPub, err = sess.DeclarePublisher(aiRespTopic); err != nil {
		log.Warn("failed to declare AI status response publisher", zap.Error(err))
		c.aiRespPub = nil
	}
	if c.aiReqSub, err = sess.DeclareSubscriber(aiReqTopic, c.onAIStatusRequest); err != nil {
		log.Warn("failed to subscribe to AI status requests", zap.Error(err))
		c.aiReqSub = nil
	}

	if c.mode == "guard" {
		c.guard = newGuardWatcher(util.StringFrom(cfg["face_presence_base_url"], ""), log)
	}

	log.Info("connector initialized",
		zap.String("cmd_vel_topic", cmdVelTopic), zap.String("mode", c.mode))
	return c, nil
}

func (c *moveConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("move_go2_autonomy: unexpected input type %T", input)
	}
	action, _ := args["action"].(string)

	c.log.Info("AI command", zap.String("action", action))

	if c.mode == "guard" && c.guard != nil && c.guard.unknownFaces() > 0 {
		c.log.Info("guard mode active and unknown face detected - ignoring command")
		return nil, nil
	}

	if !c.aiControlEnabled.Load() {
		c.log.Info("AI control disabled - ignoring command")
		return nil, nil
	}

	pos := c.odom.Position()

	if pos.Moving {
		c.log.Info("robot already moving - ignoring command")
		return nil, nil
	}

	c.mu.Lock()
	hasPending := c.pending != nil
	c.mu.Unlock()
	if hasPending {
		c.log.Info("movement in progress - ignoring command")
		return nil, nil
	}

	if pos.OdomX == 0.0 {
		c.log.Info("waiting for location data")
		return nil, nil
	}

	switch action {
	case "turn left":
		c.queuePathMove(pos, c.paths.Movement().TurnLeft, "turn left")
	case "turn right":
		c.queuePathMove(pos, c.paths.Movement().TurnRight, "turn right")
	case "move forwards":
		c.queuePathMove(pos, c.paths.Movement().Advance, "advance")
	case "move back":
		c.processMoveBack(pos)
	case "stand still":
		c.log.Info("stand still")
	default:
		c.log.Info("unknown command", zap.String("action", action))
	}

	return nil, nil
}

// queuePathMove picks a random safe path from options, then queues a movement command.
func (c *moveConnector) queuePathMove(pos go2.OdomPosition, options []uint32, label string) {
	if len(options) == 0 {
		c.log.Warn("cannot " + label + " due to barrier")
		return
	}

	angle := pathAngles[options[c.rng.Intn(len(options))]]

	c.queue(&moveCommand{
		dx:           0.5,
		yaw:          normalizeAngle(-pos.OdomYawM180P180 + angle),
		startX:       pos.OdomX,
		startY:       pos.OdomY,
		turnComplete: angle == 0,
		speed:        moveSpeed,
	})
}

// processMoveBack queues a straight retreat with no turning phase.
func (c *moveConnector) processMoveBack(pos go2.OdomPosition) {
	if !c.paths.Movement().Retreat {
		c.log.Warn("cannot retreat due to barrier")
		return
	}

	c.queue(&moveCommand{
		dx:           -0.5,
		yaw:          0.0,
		startX:       pos.OdomX,
		startY:       pos.OdomY,
		turnComplete: true,
		speed:        0.2,
	})
}

// queue installs cmd as the pending movement and resets the convergence counters.
func (c *moveConnector) queue(cmd *moveCommand) {
	c.mu.Lock()
	c.pending = cmd
	c.movementAttempts = 0
	c.gapPrevious = 0
	c.mu.Unlock()
}

// Tick advances the active movement command one step per runtime cycle.
func (c *moveConnector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
		return
	case <-time.After(tickInterval):
	}

	pos := c.odom.Position()

	if pos.OdomX == 0.0 {
		return
	}
	if pos.BodyAttitude != go2.RobotStateStanding {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pending == nil {
		return
	}

	if c.movementAttempts > movementAttemptLimit {
		c.log.Info("not converging - aborting", zap.Int("attempts", c.movementAttempts))
		c.abortLocked()
		return
	}

	if !c.pending.turnComplete {
		c.tickTurn(pos)
		return
	}
	c.tickDrive(pos)
}

// tickTurn runs phase 1: rotate until the robot faces the command's target yaw.
func (c *moveConnector) tickTurn(pos go2.OdomPosition) {
	gap := angleGap(-pos.OdomYawM180P180, c.pending.yaw)
	c.gapPrevious = gap

	switch {
	case math.Abs(gap) > 10.0:
		c.movementAttempts++
		if !c.executeTurn(gap) {
			c.abortLocked()
		}
	case math.Abs(gap) > angleToleranceDeg:
		c.movementAttempts++
		c.moveRobot(pos, 0, 0, math.Copysign(0.2, gap))
	default:
		c.log.Info("turn complete, starting advance")
		c.pending.turnComplete = true
		c.gapPrevious = 0
	}
}

// tickDrive runs phase 2: drive forwards/backwards until the target distance is reached.
func (c *moveConnector) tickDrive(pos go2.OdomPosition) {
	if c.pending.dx == 0 {
		c.log.Info("no advance required, command complete")
		c.abortLocked()
		return
	}

	traveled := math.Hypot(pos.OdomX-c.pending.startX, pos.OdomY-c.pending.startY)
	gap := math.Abs(c.pending.dx) - traveled
	c.gapPrevious = math.Abs(gap)

	move := c.paths.Movement()
	var fb float64
	switch {
	case c.pending.dx > 0:
		if !slices.Contains(move.Advance, advanceClearPath) {
			c.log.Warn("cannot advance due to barrier")
			c.abortLocked()
			return
		}
		fb = 1
	case c.pending.dx < 0:
		if !move.Retreat {
			c.log.Warn("cannot retreat due to barrier")
			c.abortLocked()
			return
		}
		fb = -1
	}

	if math.Abs(gap) <= distanceTolerance {
		c.log.Info("advance complete")
		c.abortLocked()
		return
	}

	c.movementAttempts++
	if traveled < math.Abs(c.pending.dx) {
		c.moveRobot(pos, fb*c.pending.speed, 0, 0)
	} else {
		// Overshoot: nudge back the other way.
		c.moveRobot(pos, -fb*0.2, 0, 0)
	}
}

// executeTurn issues a turn command in the direction indicated by gap's sign.
func (c *moveConnector) executeTurn(gap float64) bool {
	move := c.paths.Movement()

	// left turn
	if gap > 0 {
		if len(move.TurnLeft) == 0 {
			c.log.Warn("cannot turn left due to barrier")
			return false
		}
		sharpness := float64(slices.Min(move.TurnLeft))
		c.moveRobot(c.odom.Position(), sharpness*0.15, 0, turnSpeed)
		return true
	}

	// turn right
	if len(move.TurnRight) == 0 {
		c.log.Warn("cannot turn right due to barrier")
		return false
	}
	sharpness := float64(8 - slices.Max(move.TurnRight))
	c.moveRobot(c.odom.Position(), sharpness*0.15, 0, -turnSpeed)
	return true
}

// moveRobot publishes a Twist velocity command on /cmd_vel, but only while the robot is standing.
func (c *moveConnector) moveRobot(pos go2.OdomPosition, vx, vy, vturn float64) {
	if pos.BodyAttitude != go2.RobotStateStanding {
		return
	}
	if c.cmdVel == nil {
		return
	}

	c.log.Info("cmd_vel",
		zap.Float64("vx", vx), zap.Float64("vy", vy), zap.Float64("vturn", vturn))

	payload := serializeTwist(vx, vy, 0, 0, 0, vturn)
	if err := c.cmdVel.Put(payload); err != nil {
		c.log.Error("cmd_vel put failed", zap.Error(err))
	}
}

// abortLocked stops the robot and clears the pending command.
func (c *moveConnector) abortLocked() {
	c.movementAttempts = 0
	c.gapPrevious = 0
	c.pending = nil

	if c.cmdVel != nil {
		if err := c.cmdVel.Put(serializeTwist(0, 0, 0, 0, 0, 0)); err != nil {
			c.log.Error("cmd_vel stop failed", zap.Error(err))
		}
	}
}

// onAIStatusRequest handles AI control enable/disable/status requests and
// publishes the corresponding response.
func (c *moveConnector) onAIStatusRequest(data []byte) {
	if c.aiRespPub == nil {
		return
	}

	req, err := deserializeAIStatusRequest(data)
	if err != nil {
		c.log.Error("failed to decode AI status request", zap.Error(err))
		return
	}

	switch req.code {
	case aiCodeEnabled:
		c.aiControlEnabled.Store(true)
		c.log.Info("AI control enabled")
	case aiCodeDisabled:
		c.aiControlEnabled.Store(false)
		c.log.Info("AI control disabled")
	case aiCodeStatus:
	default:
		return
	}

	code := aiCodeDisabled
	status := "AI Control Disabled"
	if c.aiControlEnabled.Load() {
		code = aiCodeEnabled
		status = "AI Control Enabled"
	}

	payload := serializeAIStatusResponse(req.frameID, req.requestID, code, status)
	if err := c.aiRespPub.Put(payload); err != nil {
		c.log.Error("AI status response put failed", zap.Error(err))
	}
}

// Stop releases all Zenoh resources and the guard watcher.
func (c *moveConnector) Stop() {
	if c.guard != nil {
		c.guard.stop()
		c.guard = nil
	}
	if c.aiReqSub != nil {
		c.aiReqSub.Drop()
		c.aiReqSub = nil
	}
	if c.aiRespPub != nil {
		c.aiRespPub.Drop()
		c.aiRespPub = nil
	}
	if c.cmdVel != nil {
		c.cmdVel.Drop()
		c.cmdVel = nil
	}
	if c.session != nil {
		c.session.Close()
		c.session = nil
	}
	c.log.Info("connector stopped")
}

// normalizeAngle wraps a degree value into the (-180, 180] range using a single
// fold, matching the Python connector's behaviour.
func normalizeAngle(angle float64) float64 {
	if angle < -180 {
		return angle + 360
	}
	if angle > 180 {
		return angle - 360
	}
	return angle
}

// angleGap returns the shortest signed angular distance (deg) from target to
// current, wrapped into [-180, 180].
func angleGap(current, target float64) float64 {
	gap := current - target
	if gap > 180 {
		return gap - 360
	}
	if gap < -180 {
		return gap + 360
	}
	return gap
}
