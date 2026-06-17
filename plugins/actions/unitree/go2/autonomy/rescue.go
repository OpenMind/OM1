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

// rescueRecenterDistanceM is the short goal distance for re-centering turns (m): the
// robot turns while barely advancing. Topic/distance constants and serializePose are
// reused from mppi.go (same package).
const rescueRecenterDistanceM = 0.3

type RescueAction string

func (RescueAction) EnumValues() []string {
	return []string{
		"turn left",
		"turn right",
		"turn left slightly",
		"turn right slightly",
		"move forwards",
		"move back",
		"stand still",
	}
}

type RescueInput struct {
	Action RescueAction `json:"action" description:"The movement to perform"`
}

func init() {
	actions.RegisterInterface(
		"unitree_go2_rescue",
		"Action interface for the Unitree Go2 laydown-rescue behavior, handing goals to the "+
			"om_mppi planner. Supported movements: turn left, turn right, turn left slightly, "+
			"turn right slightly, move forwards, move back, stand still.",
		RescueInput{},
	)
	actions.Register("unitree_go2_rescue/mppi", NewRescueConnector)
}

// RescueConnector drives the Go2 by handing goals to the om_mppi planner.
type RescueConnector struct {
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

	// lastAlertMoveSeq gates the dog to one move per VLM verdict during an alert.
	lastAlertMoveSeq atomic.Uint64

	// geometryDriven, when set, makes the connector approach a downed person directly
	// from bbox geometry during an alert, ignoring LLM movement commands.
	geometryDriven bool
	lockWidthFrac  float64
	centerTol      float64

	// lastGeoSeq gates the geometric self-drive to one move per perception frame.
	lastGeoSeq atomic.Uint64
}

// NewRescueConnector builds the MPPI-backed autonomy connector from its config.
func NewRescueConnector(cfg map[string]any) (actions.Connector, error) {
	log := logger.Get().Named("unitree_go2_rescue/mppi")

	goalTopic := util.StringFrom(cfg["goal_topic"], defaultGoalTopic)
	statusTopic := util.StringFrom(cfg["status_topic"], defaultStatusTopic)
	aiReqTopic := util.StringFrom(cfg["ai_request_topic"], defaultAIRequestTopic)
	aiRespTopic := util.StringFrom(cfg["ai_response_topic"], defaultAIRespTopic)

	c := &RescueConnector{
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
	c.geometryDriven = util.BoolFrom(cfg["geometry_driven"], false)
	c.lockWidthFrac = providers.DefaultLockWidthFrac
	if v, ok := cfg["lock_width_frac"].(float64); ok && v > 0 {
		c.lockWidthFrac = v
	}
	c.centerTol = providers.DefaultCenterTol
	if v, ok := cfg["center_tolerance"].(float64); ok && v > 0 {
		c.centerTol = v
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
		zap.Bool("geometry_driven", c.geometryDriven),
		zap.Float64("lock_width_frac", c.lockWidthFrac),
		zap.Float64("center_tolerance", c.centerTol),
		zap.String("mode", c.mode))
	return c, nil
}

// Connect accepts AI commands to move the robot, translating them into MPPI goals and publishing on the goal topic.
func (c *RescueConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unitree_go2_rescue/mppi: unexpected input type %T", input)
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

	if c.geometryDriven && providers.PersonDownAlert() {
		c.log.Info("geometry-driven approach active - ignoring LLM movement during alert",
			zap.String("action", action))
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

	if providers.PersonDownArrived() {
		c.mu.Lock()
		active := c.active
		c.mu.Unlock()
		if active {
			c.cancelGoal()
		}
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

// issueGoal selects a path (smallest-magnitude heading when gentlest, else random)
// and publishes a goal toward it. pathAngles is +right but the mppi goal frame is
// +left (CCW), so the angle is negated.
func (c *RescueConnector) issueGoal(options []uint32, label string, gentlest bool) {
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
func (c *RescueConnector) issueRotation(options []uint32, label string) {
	if len(options) == 0 {
		c.log.Warn("cannot " + label + " due to barrier")
		return
	}
	angleRad := -pathAngles[gentlestPath(options)] * math.Pi / 180.0
	if angleRad == 0 {
		c.log.Info("already centered - holding")
		return
	}
	bx := rescueRecenterDistanceM * math.Cos(angleRad)
	by := rescueRecenterDistanceM * math.Sin(angleRad)
	if err := c.publishGoal(bx, by, angleRad); err != nil {
		return
	}
	c.markActive()
	c.log.Info("issued mppi recenter goal", zap.String("label", label),
		zap.Float64("goal_x", bx), zap.Float64("goal_y", by))
}

// gentlestPath returns the option whose heading is closest to straight ahead.
func gentlestPath(options []uint32) uint32 {
	best := options[0]
	for _, o := range options[1:] {
		if math.Abs(pathAngles[o]) < math.Abs(pathAngles[best]) {
			best = o
		}
	}
	return best
}

// issueRetreat publishes a goal a short distance straight behind the robot.
func (c *RescueConnector) issueRetreat(allowed bool) {
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
func (c *RescueConnector) publishGoal(x, y, yaw float64) error {
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
func (c *RescueConnector) publishReverseGoal(distance float64) error {
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
func (c *RescueConnector) cancelGoal() {
	_ = c.publishGoal(0, 0, 0)
	c.mu.Lock()
	c.active = false
	c.mu.Unlock()
}

// markActive records that a goal is in flight, for gating and timeout.
func (c *RescueConnector) markActive() {
	c.mu.Lock()
	c.active = true
	c.issuedAt = time.Now()
	c.mu.Unlock()
}

// onStatus consumes om_mppi status updates; a terminal status frees the
// connector to accept the next AI command.
func (c *RescueConnector) onStatus(data []byte) {
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
func (c *RescueConnector) Tick(ctx context.Context) {
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

	if c.geometryDriven {
		c.geometryDrive()
	}
}

// geometryDrive approaches a downed person directly from bbox geometry while an alert
// is active: it re-centers the target with a slight turn, steps forward once centered,
// and locks on (holding position) once the target fills enough of the frame. It runs
// at the Tick cadence, fully decoupled from the LLM, but reuses the same odom/barrier/
// gating guards as LLM-driven moves and acts at most once per fresh perception frame.
func (c *RescueConnector) geometryDrive() {
	if !providers.PersonDownAlert() || providers.PersonDownArrived() {
		return
	}
	target := providers.FallenTargetSnapshot()
	if !target.Present {
		return
	}

	seq := providers.VisionSeq()
	if seq == c.lastGeoSeq.Load() {
		return
	}

	c.mu.Lock()
	busy := c.active
	c.mu.Unlock()
	if busy {
		return
	}

	pos := c.odom.Position()
	if pos.Moving {
		return
	}
	if pos.OdomX == 0.0 {
		c.log.Debug("bbox approach: waiting for location data")
		return
	}

	c.lastGeoSeq.Store(seq)

	move := c.paths.Movement()
	switch approachDecision(target.NormErrX, target.WidthFrac, c.lockWidthFrac, c.centerTol) {
	case approachLock:
		providers.SetPersonDownArrived(true)
		c.cancelGoal()
		c.log.Info("bbox approach: locked on - holding position",
			zap.Float64("width_frac", target.WidthFrac),
			zap.Float64("lock_width_frac", c.lockWidthFrac))
	case approachRecenterRight:
		c.issueRotation(move.TurnRight, "bbox recenter right")
	case approachRecenterLeft:
		c.issueRotation(move.TurnLeft, "bbox recenter left")
	case approachAdvance:
		c.issueGoal(move.Advance, "bbox advance", false)
	}
}

type approachAction int

const (
	approachAdvance approachAction = iota
	approachRecenterLeft
	approachRecenterRight
	approachLock
)

// approachDecision maps the target's bbox geometry to the next approach move:
// lock on once close enough (wide bbox), else re-center if off to a side, else step
// forward. Lock is checked first so a centered-and-near target stops rather than steps.
func approachDecision(normErrX, widthFrac, lockWidthFrac, centerTol float64) approachAction {
	switch {
	case widthFrac >= lockWidthFrac:
		return approachLock
	case normErrX > centerTol:
		return approachRecenterRight
	case normErrX < -centerTol:
		return approachRecenterLeft
	default:
		return approachAdvance
	}
}

// onAIStatusRequest handles AI control enable/disable/status requests and
// publishes the corresponding response (mirrors move.go).
func (c *RescueConnector) onAIStatusRequest(data []byte) {
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
func (c *RescueConnector) Stop() {
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
