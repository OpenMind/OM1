package autonomy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
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
	defaultGoalTopic   = "om/mppi/goal"
	defaultStatusTopic = "om/mppi/status"

	mppiGoalDistanceM     = 1.2 // goal distance ahead along the chosen path (m); under the 2.0 m om_path length
	mppiReverseDistanceM  = 0.6 // distance to back straight up on "move back" (m)
	mppiRecenterDistanceM = 0.3 // short goal distance for re-centering turns (m); turns while barely advancing
)

func init() {
	actions.Register("unitree_go2_autonomy/mppi", NewMPPIConnector)
}

// MPPIConnector drives the Go2 by handing goals to the om_mppi planner.
type MPPIConnector struct {
	log *zap.Logger

	odom  *go2.OdomZenohProvider
	paths *providers.PathsProvider

	session   zenohsession.Session
	goalPub   zenohsession.Publisher
	statusSub zenohsession.Subscriber

	aiRespPub zenohsession.Publisher
	aiReqSub  zenohsession.Subscriber

	aiControlEnabled atomic.Bool

	mode        string
	gentleTurns bool
	guard       *guardWatcher

	rng *rand.Rand

	goalDistance float64

	mu       sync.Mutex
	active   bool
	issuedAt time.Time

	minActiveHold time.Duration
	lastAction    string

	// lastAlertMoveSeq is the VLM verdict (vision sequence) the most recent move acted
	// on while a person-down alert was active. It gates the dog to one move per fresh
	// verdict during an alert, so it holds instead of acting on the stale latched
	// verdict while the multi-second VLM call computes the next one.
	lastAlertMoveSeq atomic.Uint64
}

// NewMPPIConnector builds the MPPI-backed autonomy connector from its config.
func NewMPPIConnector(cfg map[string]any) (actions.Connector, error) {
	log := logger.Get().Named("unitree_go2_autonomy/mppi")

	goalTopic := util.StringFrom(cfg["goal_topic"], defaultGoalTopic)
	statusTopic := util.StringFrom(cfg["status_topic"], defaultStatusTopic)
	aiReqTopic := util.StringFrom(cfg["ai_request_topic"], defaultAIRequestTopic)
	aiRespTopic := util.StringFrom(cfg["ai_response_topic"], defaultAIRespTopic)

	c := &MPPIConnector{
		log:           log,
		odom:          go2.OdomZenoh(),
		paths:         providers.NewPathsProvider(),
		mode:          util.StringFrom(cfg["mode"], ""),
		gentleTurns:   util.BoolFrom(cfg["gentle_turns"], false),
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
		goalDistance:  mppiGoalDistanceM,
		minActiveHold: 3 * time.Second,
	}
	if d, ok := cfg["goal_distance"].(float64); ok && d > 0 {
		c.goalDistance = d
	}
	if h, ok := cfg["min_active_hold_seconds"].(float64); ok && h >= 0 {
		c.minActiveHold = time.Duration(h * float64(time.Second))
	}
	c.aiControlEnabled.Store(true)

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn("zenoh unavailable, movement disabled", zap.Error(err))
		return c, nil
	}
	c.session = sess

	if c.goalPub, err = sess.DeclarePublisher(goalTopic); err != nil {
		log.Warn("failed to declare goal publisher, movement disabled", zap.Error(err))
		c.goalPub = nil
	}
	if c.statusSub, err = sess.DeclareSubscriber(statusTopic, c.onStatus); err != nil {
		log.Warn("failed to subscribe to mppi status", zap.Error(err))
		c.statusSub = nil
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

	log.Info("mppi connector initialized",
		zap.String("goal_topic", goalTopic),
		zap.String("status_topic", statusTopic),
		zap.Float64("goal_distance", c.goalDistance),
		zap.String("mode", c.mode))
	return c, nil
}

// Connect accepts AI commands to move the robot, translating them into MPPI goals and publishing on the goal topic.
func (c *MPPIConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unitree_go2_autonomy/mppi: unexpected input type %T", input)
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

	if action == "stand still" {
		c.mu.Lock()
		active := c.active
		runFor := time.Since(c.issuedAt)
		lastAction := c.lastAction
		c.lastAction = action
		c.mu.Unlock()

		deliberate := !active ||
			runFor >= c.minActiveHold ||
			lastAction == "stand still"
		if deliberate {
			c.cancelGoal()
			c.log.Info("stand still")
		} else {
			c.log.Info("ignoring stand still - active goal still progressing",
				zap.Duration("run_for", runFor),
				zap.Duration("min_hold", c.minActiveHold))
		}
		return nil, nil
	}

	// Locked on: once we have arrived and centered on a downed person, hold position
	// for the rest of the emergency and ignore every move command. The VLM Location is
	// jittery, so re-centering now over-rotates and loses the person. Unlatches when
	// the alert clears. ("stand still" was already handled above.)
	if providers.PersonDownArrived() {
		c.log.Info("arrived - locked on, holding position", zap.String("action", action))
		return nil, nil
	}

	c.mu.Lock()
	c.lastAction = action
	busy := c.active
	c.mu.Unlock()
	if busy {
		c.log.Info("movement in progress - ignoring command")
		return nil, nil
	}

	pos := c.odom.Position()
	if pos.Moving {
		c.log.Info("robot already moving - ignoring command")
		return nil, nil
	}
	if pos.OdomX == 0.0 {
		c.log.Info("waiting for location data")
		return nil, nil
	}

	// During a person-down alert, move at most once per fresh VLM verdict. The cortex
	// re-serves the latched verdict on stale timer ticks while the multi-second VLM
	// call computes the next one; without this gate the dog keeps moving on stale
	// perception. Hold (do nothing) until a new verdict arrives. Patrol is unaffected.
	if providers.PersonDownAlert() {
		seq := providers.VisionSeq()
		if seq == c.lastAlertMoveSeq.Load() {
			c.log.Info("holding for fresh vision frame", zap.String("action", action))
			return nil, nil
		}
		c.lastAlertMoveSeq.Store(seq)
	}

	move := c.paths.Movement()
	switch action {
	case "turn left":
		c.issueGoal(move.TurnLeft, "turn left", c.gentleTurns || providers.PersonDownAlert())
	case "turn right":
		c.issueGoal(move.TurnRight, "turn right", c.gentleTurns || providers.PersonDownAlert())
	case "turn left slightly":
		c.issueRotation(move.TurnLeft, "turn left slightly")
	case "turn right slightly":
		c.issueRotation(move.TurnRight, "turn right slightly")
	case "move forwards":
		c.issueGoal(move.Advance, "advance", false)
	case "move back":
		c.issueRetreat(move.Retreat)
	default:
		c.log.Info("unknown command", zap.String("action", action))
	}

	return nil, nil
}

// issueGoal selects a path from the provided indices and publishes a goal toward
// it: the smallest-magnitude heading when gentlest, else a random one.
func (c *MPPIConnector) issueGoal(options []uint32, label string, gentlest bool) {
	if len(options) == 0 {
		c.log.Warn("cannot " + label + " due to barrier")
		return
	}

	var chosen uint32
	if gentlest {
		chosen = gentlestPath(options)
	} else {
		chosen = options[c.rng.Intn(len(options))]
	}
	// pathAngles is +right (TurnRight = +15..+60), but the mppi goal frame is +left
	// (CCW, standard ROS) for both y and yaw — confirmed by odom: a +yaw goal drives
	// odom yaw_m180_p180 positive, i.e. left. So negate to make "turn right" go right.
	angleRad := -pathAngles[chosen] * math.Pi / 180.0
	bx := c.goalDistance * math.Cos(angleRad)
	by := c.goalDistance * math.Sin(angleRad)
	if err := c.publishGoal(bx, by, angleRad); err != nil {
		return
	}
	c.markActive()
	c.log.Info("issued mppi goal",
		zap.String("label", label),
		zap.Float64("goal_x", bx), zap.Float64("goal_y", by))
}

// issueRotation re-centers a target by turning toward the gentlest safe heading
// on the chosen side. It uses a short goal distance (a small arc) so the robot
// turns while barely advancing — the planner won't execute a zero-translation
// goal, so the goal must move a little.
func (c *MPPIConnector) issueRotation(options []uint32, label string) {
	if len(options) == 0 {
		c.log.Warn("cannot " + label + " due to barrier")
		return
	}
	// Negate: pathAngles is +right, mppi goal frame is +left (CCW). See issueGoal.
	// (One-move-per-verdict gating during an alert is handled in Connect.)
	angleRad := -pathAngles[gentlestPath(options)] * math.Pi / 180.0
	if angleRad == 0 {
		c.log.Info("already centered - holding")
		return
	}
	bx := mppiRecenterDistanceM * math.Cos(angleRad)
	by := mppiRecenterDistanceM * math.Sin(angleRad)
	if err := c.publishGoal(bx, by, angleRad); err != nil {
		return
	}
	c.markActive()
	c.log.Info("issued mppi recenter goal", zap.String("label", label),
		zap.Float64("goal_x", bx), zap.Float64("goal_y", by))
}

// issueRetreat publishes a goal a short distance straight behind the robot.
func (c *MPPIConnector) issueRetreat(allowed bool) {
	if !allowed {
		c.log.Warn("cannot retreat due to barrier")
		return
	}
	if err := c.publishReverseGoal(mppiReverseDistanceM); err != nil {
		return
	}
	c.markActive()
	c.log.Info("issued mppi reverse goal")
}

// publishGoal serialises and sends a body-frame goal pose to om_mppi.
func (c *MPPIConnector) publishGoal(x, y, yaw float64) error {
	if c.goalPub == nil {
		return fmt.Errorf("goal publisher unavailable")
	}
	if err := c.goalPub.Put(serializePose(x, y, yaw, false)); err != nil {
		c.log.Error("goal put failed", zap.Error(err))
		return err
	}
	return nil
}

// publishReverseGoal sends a goal that commands the robot to back up in a straight line.
func (c *MPPIConnector) publishReverseGoal(distance float64) error {
	if c.goalPub == nil {
		return fmt.Errorf("goal publisher unavailable")
	}
	if err := c.goalPub.Put(serializePose(-distance, 0, 0, true)); err != nil {
		c.log.Error("reverse goal put failed", zap.Error(err))
		return err
	}
	return nil
}

// cancelGoal stops the planner by sending a goal at the origin and clears state.
func (c *MPPIConnector) cancelGoal() {
	_ = c.publishGoal(0, 0, 0)
	c.mu.Lock()
	c.active = false
	c.mu.Unlock()
}

// markActive records that a goal is in flight, for gating and timeout.
func (c *MPPIConnector) markActive() {
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now()
	c.mu.Unlock()
}

// onStatus consumes om_mppi status updates; a terminal status frees the
// connector to accept the next AI command.
func (c *MPPIConnector) onStatus(data []byte) {
	status, _, err := readCDRString(data, 4, false)
	if err != nil {
		c.log.Error("failed to decode mppi status", zap.Error(err))
		return
	}

	switch status {
	case "reached", "idle", "blocked":
		c.mu.Lock()
		c.active = false
		c.mu.Unlock()
	}
	c.log.Debug("mppi status", zap.String("status", status))
}

// Tick is a safety backstop: if a goal never reports completion, clear it after
// commandTimeout so the robot does not get stuck on a stale command.
func (c *MPPIConnector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
		return
	case <-time.After(tickInterval):
	}

	c.mu.Lock()
	stale := c.active && time.Since(c.issuedAt) > commandTimeout
	if stale {
		c.active = false
	}
	c.mu.Unlock()

	if stale {
		c.log.Info("mppi goal timeout - clearing and stopping")
		_ = c.publishGoal(0, 0, 0)
	}
}

// onAIStatusRequest handles AI control enable/disable/status requests and
// publishes the corresponding response (mirrors move.go).
func (c *MPPIConnector) onAIStatusRequest(data []byte) {
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
func (c *MPPIConnector) Stop() {
	if c.guard != nil {
		c.guard.stop()
		c.guard = nil
	}
	if c.statusSub != nil {
		c.statusSub.Drop()
		c.statusSub = nil
	}
	if c.aiReqSub != nil {
		c.aiReqSub.Drop()
		c.aiReqSub = nil
	}
	if c.aiRespPub != nil {
		c.aiRespPub.Drop()
		c.aiRespPub = nil
	}
	if c.goalPub != nil {
		c.goalPub.Drop()
		c.goalPub = nil
	}
	if c.session != nil {
		c.session.Close()
		c.session = nil
	}
	c.log.Info("mppi connector stopped")
}

// serializePose encodes a geometry_msgs/Pose in CDR little-endian format, ready
// to publish on a ROS2 topic.  Yaw is encoded as a planar quaternion.
//
// Wire layout (absolute offsets from start of buffer):
//
//	[0]  CDR encapsulation header: 0x00 0x01 0x00 0x00
//	[4]  position.x     float64 LE
//	[12] position.y     float64 LE
//	[20] position.z     float64 LE
//	[28] orientation.x  float64 LE
//	[36] orientation.y  float64 LE
//	[44] orientation.z  float64 LE
//	[52] orientation.w  float64 LE
//
// Every field is a float64 on an 8-byte aligned data offset, so no interior
// padding is required.
func serializePose(x, y, yaw float64, reverse bool) []byte {
	buf := make([]byte, 0, 4+7*8)

	z := 0.0
	if reverse {
		z = 1.0
	}

	buf = append(buf, 0x00, 0x01, 0x00, 0x00)
	buf = zenohsession.AppendFloat64LE(buf, x)
	buf = zenohsession.AppendFloat64LE(buf, y)
	buf = zenohsession.AppendFloat64LE(buf, z)
	buf = zenohsession.AppendFloat64LE(buf, 0)
	buf = zenohsession.AppendFloat64LE(buf, 0)
	buf = zenohsession.AppendFloat64LE(buf, math.Sin(yaw/2))
	buf = zenohsession.AppendFloat64LE(buf, math.Cos(yaw/2))

	return buf
}
