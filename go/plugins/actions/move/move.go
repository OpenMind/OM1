// Package move registers the "move" action plugin.
//
// To add a new connector, implement actions.Connector and call
// actions.Register("move/<type>", factory) in init().
package move

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
)

func init() {
	// Declare the input schema once; all connectors for "move" share it.
	actions.RegisterInterface(
		"move",
		"Action interface for robot movement commands. "+
			"Enables the robot to perform various predefined movement behaviours "+
			"such as standing still, sitting, walking, running, and jumping.",
		MoveInput{},
	)

	actions.Register("move/passthrough", newPassthrough)
	actions.Register("move/ros2", newROS2)
}

// ─── Input type ──────────────────────────────────────────────────────────────

// MovementAction is the set of valid movement commands.
type MovementAction string

// EnumValues implements actions.Enumer so the schema generator can enumerate
// the allowed values automatically.
func (MovementAction) EnumValues() []string {
	return []string{
		"stand still",
		"sit",
		"dance",
		"shake paw",
		"walk",
		"walk back",
		"run",
		"jump",
		"wag tail",
	}
}

// MoveInput is the structured argument the LLM passes when invoking "move".
type MoveInput struct {
	Action MovementAction `json:"action" description:"The movement action to execute"`
}

// ─── Passthrough connector (logs only) ───────────────────────────────────────

type passthrough struct {
	log *zap.Logger
}

func newPassthrough(_ map[string]any) (actions.Connector, error) {
	logger, _ := zap.NewProduction()
	return &passthrough{log: logger}, nil
}

func (p *passthrough) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	p.log.Info("move/passthrough", zap.Any("input", input))
	return nil, nil
}

func (p *passthrough) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(60 * time.Second):
	}
}
func (p *passthrough) Stop()                  {}

// ─── ROS2 connector stub ─────────────────────────────────────────────────────

type ros2Connector struct {
	// TODO: embed a ROS2 publisher client
}

func newROS2(_ map[string]any) (actions.Connector, error) {
	return &ros2Connector{}, nil
}

func (r *ros2Connector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("move/ros2: unexpected input type %T", input)
	}
	actionName, _ := arguments["action"].(string)
	_ = actionName
	// TODO: publish to /cmd_vel or equivalent ROS2 topic
	return nil, nil
}

func (r *ros2Connector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(60 * time.Second):
	}
}
func (r *ros2Connector) Stop()                  {}
