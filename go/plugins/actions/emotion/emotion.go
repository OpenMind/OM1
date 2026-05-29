package emotion

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
)

// EmotionAction is a string enum of supported facial expressions.
type EmotionAction string

func (EmotionAction) EnumValues() []string {
	return []string{"happy", "confused", "curious", "excited", "sad", "think"}
}

type EmotionInput struct {
	Action EmotionAction `json:"action" description:"The facial expression to display"`
}

func init() {
	actions.RegisterInterface(
		"emotion",
		"Action interface for robot facial expression control."+
			"This action enables the robot to display various facial expressions to convey"+
			"emotional states or cognitive states. The specific expression is determined"+
			"by the FaceAction enum value provided in the input."+
			"Supported expressions include emotional states (HAPPY, SAD, EXCITED) and"+
			"cognitive states (THINK, CURIOUS, CONFUSED), allowing the robot to provide"+
			"visual feedback that enhances human-robot interaction.",
		EmotionInput{},
	)
	actions.Register("emotion/log", newLogConnector)
}

type logConnector struct {
	log *zap.Logger
}

func newLogConnector(_ map[string]any) (actions.Connector, error) {
	log, _ := zap.NewProduction()
	return &logConnector{log: log}, nil
}

func (c *logConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, nil
	}
	emotion, _ := args["action"].(string)
	c.log.Info("emotion", zap.String("expression", emotion))
	return nil, nil
}

func (c *logConnector) Tick(_ context.Context) {}
func (c *logConnector) Stop()                   {}
