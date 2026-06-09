package go2

import (
	"context"
	"encoding/json"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const defaultFrontierExplorationTopic = "explore/status"

func init() {
	bg.Register("UnitreeGo2FrontierExploration", NewUnitreeGo2FrontierExploration)
}

type frontierExplorationConfig struct {
	Topic            string `json:"topic"`
	ContextAwareText string `json:"context_aware_text"`
}

type frontierExplorationMessage struct {
	Complete bool   `json:"complete"`
	Info     string `json:"info"`
}

type UnitreeGo2FrontierExploration struct {
	log              *zap.Logger
	session          zenohsession.Session
	sub              zenohsession.Subscriber
	contextAwareText map[string]any
}

// NewUnitreeGo2FrontierExploration creates a new UnitreeGo2FrontierExploration background task.
func NewUnitreeGo2FrontierExploration(configMap map[string]any) (bg.Background, error) {
	log := logger.Get().Named("UnitreeGo2FrontierExploration")

	cfg := frontierExplorationConfig{
		Topic:            defaultFrontierExplorationTopic,
		ContextAwareText: `{"exploration_done": true}`,
	}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.Topic == "" {
		cfg.Topic = defaultFrontierExplorationTopic
	}

	contextAwareText := map[string]any{"exploration_done": true}
	if cfg.ContextAwareText != "" {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(cfg.ContextAwareText), &parsed); err != nil {
			log.Error("error decoding context_aware_text JSON; using default", zap.Error(err))
		} else {
			contextAwareText = parsed
		}
	}

	e := &UnitreeGo2FrontierExploration{log: log, contextAwareText: contextAwareText}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Error("zenoh unavailable", zap.Error(err))
		return e, nil
	}
	e.session = sess

	sub, err := sess.DeclareSubscriber(cfg.Topic, e.onStatus)
	if err != nil {
		log.Error("failed to declare subscriber", zap.Error(err))
	} else {
		e.sub = sub
	}

	log.Info("background task initialized", zap.String("topic", cfg.Topic))
	return e, nil
}

// onStatus is called when a message is received on the status topic.
func (e *UnitreeGo2FrontierExploration) onStatus(data []byte) {
	if len(data) == 0 {
		return
	}

	message, err := zenohsession.ReadStdMsgsString(data)
	if err != nil {
		e.log.Error("failed to deserialize status payload", zap.Error(err))
		return
	}

	var msg frontierExplorationMessage
	if err := json.Unmarshal([]byte(message), &msg); err != nil {
		e.log.Error("error decoding exploration message JSON", zap.Error(err))
		return
	}

	if !msg.Complete {
		return
	}

	e.log.Info("Exploration Status: Completed", zap.String("info", msg.Info))

	providers.ModeContext().Publish(e.contextAwareText)
}

func (e *UnitreeGo2FrontierExploration) Run(ctx context.Context) {
	<-ctx.Done()
}

func (e *UnitreeGo2FrontierExploration) Stop() {
	e.log.Info("stopping background task")
	if e.sub != nil {
		e.sub.Drop()
		e.sub = nil
	}

	if e.session != nil {
		e.session.Close()
		e.session = nil
	}
}
