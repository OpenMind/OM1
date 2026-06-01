package emotion

import (
	"context"
	"fmt"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

type Emotion string

func (Emotion) EnumValues() []string {
	return []string{
		"happy",
		"confused",
		"curious",
		"excited",
		"sad",
		"think",
	}
}

type EmotionInput struct {
	Action Emotion `json:"action" description:"The facial expression to display"`
}

func init() {
	actions.RegisterInterface(
		"emotion",
		"Action interface for robot facial expression control. "+
			"Publishes avatar emotion commands via Zenoh. "+
			"Supported expressions: happy, confused, curious, excited, sad, think.",
		EmotionInput{},
	)
	actions.Register("emotion/zenoh", NewEmotionZenohConnector)
}

type zenohConnector struct {
	log      *zap.Logger
	provider *providers.AvatarProvider
}

// NewZenohConnector creates a new zenohConnector with the provided configuration.
func NewEmotionZenohConnector(cfg map[string]any) (actions.Connector, error) {
	return &zenohConnector{
		log:      logger.Get(),
		provider: providers.Avatar(),
	}, nil
}

func (z *zenohConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("emotion/zenoh: unexpected input type %T", input)
	}
	emotion, _ := args["action"].(string)
	if emotion == "" {
		return nil, nil
	}

	if err := z.provider.SendAvatarCommand(emotion); err != nil {
		z.log.Error("emotion/zenoh: send failed", zap.Error(err))
	}
	return nil, nil
}

// Tick is a no-op since this connector is event-driven.
func (z *zenohConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the AvatarProvider manages its own resources.
func (z *zenohConnector) Stop() {}
