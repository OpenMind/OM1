package navigation

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
	go2 "github.com/openmind/om1/internal/providers/unitree/go2"
	"github.com/openmind/om1/internal/util"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	simLocationsURL   = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/list"
	localLocationsURL = "http://localhost:5000/maps/locations/list"

	defaultGoalPoseTopic  = "goal_pose"
	defaultNavStatusTopic = "navigate_to_pose/_action/status"
	defaultAIStatusTopic  = "om/ai/request"

	defaultTimeout = 5 * time.Second
	defaultRefresh = 30 * time.Second

	frameMap = "map"
)

var labelPrefixes = []string{
	"go to the ",
	"go to ",
	"navigate to the ",
	"navigate to ",
	"move to the ",
	"move to ",
	"take me to the ",
	"take me to ",
}

type NavigateLocationInput struct {
	Action string `json:"action" description:"The location name to navigate to (e.g. \"kitchen\", \"sofa\", \"table\"). Use ONLY the location name — do NOT include phrases like \"go to\" or \"navigate to\". It must exactly match one of the saved location names."`
}

func init() {
	actions.RegisterInterface(
		"navigation",
		"Action interface to navigate the Unitree Go2 to a saved location by name. "+
			"The 'action' field must contain ONLY the location name (e.g. \"kitchen\", \"sofa\"), "+
			"extracted from commands like \"go to the kitchen\" or \"take me to the sofa\". "+
			"It publishes a Nav2 goal pose on the /goal_pose Zenoh topic and disables AI "+
			"movement control until navigation completes.",
		NavigateLocationInput{},
	)
	actions.Register("navigation/navigate", NewUnitreeGo2Navigate)
}

type Config struct {
	BaseURL         string `json:"base_url"`
	APIKey          string `json:"api_key"`
	UseSim          bool   `json:"use_sim"`
	Timeout         int    `json:"timeout"`
	RefreshInterval int    `json:"refresh_interval"`

	GoalPoseTopic  string `json:"goal_pose_topic"`
	NavStatusTopic string `json:"nav_status_topic"`
	AIStatusTopic  string `json:"ai_status_topic"`

	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
}

// parseConfig decodes the raw config map and applies defaults.
func parseConfig(configMap map[string]any) Config {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	if cfg.BaseURL == "" {
		if cfg.UseSim {
			cfg.BaseURL = simLocationsURL
		} else {
			cfg.BaseURL = localLocationsURL
		}
	}
	if cfg.GoalPoseTopic == "" {
		cfg.GoalPoseTopic = defaultGoalPoseTopic
	}
	if cfg.NavStatusTopic == "" {
		cfg.NavStatusTopic = defaultNavStatusTopic
	}
	if cfg.AIStatusTopic == "" {
		cfg.AIStatusTopic = defaultAIStatusTopic
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = tts.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = tts.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = tts.DefaultRate
	}

	return cfg
}

// Connector implements the actions.Connector interface for navigating the Unitree Go2 to saved locations.
type Connector struct {
	log       *zap.Logger
	tts       *tts.ElevenLabsProvider
	locations *go2.LocationsProvider

	session      zenohsession.Session
	goalPosePub  zenohsession.Publisher
	aiStatusPub  zenohsession.Publisher
	navStatusSub zenohsession.Subscriber

	navInProgress atomic.Bool

	mu          sync.Mutex
	destination string
}

// NewUnitreeGo2Navigate builds the connector from a decoded config map.
func NewUnitreeGo2Navigate(configMap map[string]any) (actions.Connector, error) {
	cfg := parseConfig(configMap)

	timeout := defaultTimeout
	if cfg.Timeout > 0 {
		timeout = time.Duration(cfg.Timeout) * time.Second
	}
	refresh := defaultRefresh
	if cfg.RefreshInterval > 0 {
		refresh = time.Duration(cfg.RefreshInterval) * time.Second
	}

	log := logger.Get().Named("unitree_go2_navigate/navigate")

	elevenlabs := tts.ElevenLabs(tts.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	locations := go2.NewLocationsProvider(cfg.BaseURL, cfg.APIKey, timeout, refresh)
	locations.Start()

	c := &Connector{
		log:       log,
		tts:       elevenlabs,
		locations: locations,
	}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Warn("zenoh unavailable, navigation disabled", zap.Error(err))
		return c, nil
	}
	c.session = sess

	if c.goalPosePub, err = sess.DeclarePublisher(cfg.GoalPoseTopic); err != nil {
		log.Warn("failed to declare goal_pose publisher, navigation disabled", zap.Error(err))
		c.goalPosePub = nil
	}
	if c.aiStatusPub, err = sess.DeclarePublisher(cfg.AIStatusTopic); err != nil {
		log.Warn("failed to declare AI status publisher", zap.Error(err))
		c.aiStatusPub = nil
	}
	if c.navStatusSub, err = sess.DeclareSubscriber(cfg.NavStatusTopic, c.onNavStatus); err != nil {
		log.Warn("failed to subscribe to navigation status", zap.Error(err))
		c.navStatusSub = nil
	}

	log.Info("connector initialized",
		zap.String("base_url", cfg.BaseURL),
		zap.String("goal_pose_topic", cfg.GoalPoseTopic),
		zap.String("nav_status_topic", cfg.NavStatusTopic))
	return c, nil
}

// Connect resolves the requested location and publishes a Nav2 goal pose.
func (c *Connector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unitree_go2_navigate/navigate: unexpected input type %T", input)
	}

	label := cleanLabel(args["action"])
	if label == "" {
		return nil, nil
	}

	loc, found := c.locations.GetLocation(label)
	if !found {
		names := c.locations.AllNames()
		if len(names) > 0 {
			c.log.Warn("location not found",
				zap.String("label", label),
				zap.String("available", strings.Join(names, ", ")))
		} else {
			c.log.Warn("location not found and no locations available", zap.String("label", label))
		}
		return nil, nil
	}

	if c.goalPosePub == nil {
		c.log.Error("cannot publish goal pose; zenoh publisher unavailable")
		return nil, nil
	}

	c.mu.Lock()
	c.destination = label
	c.mu.Unlock()

	if !c.navInProgress.Load() {
		c.publishAIStatus(false)
	}
	c.navInProgress.Store(true)

	payload := serializePoseStamped(frameMap, loc.Pose)
	if err := c.goalPosePub.Put(payload); err != nil {
		c.log.Error("goal_pose put failed", zap.Error(err))
		return nil, nil
	}

	c.log.Info("navigation initiated", zap.String("label", label))
	return nil, nil
}

func (c *Connector) onNavStatus(data []byte) {
	code, ok := parseNav2StatusLatest(data)
	if !ok {
		return
	}

	c.log.Info("navigation status", zap.String("status", statusName(code)), zap.Int32("code", code))

	switch code {
	case statusAccepted, statusExecuting:
		if !c.navInProgress.Load() {
			c.navInProgress.Store(true)
			c.publishAIStatus(false)
			c.log.Info("navigation started - AI control disabled")
		}
	case statusSucceeded:
		if c.navInProgress.Swap(false) {
			c.publishAIStatus(true)
			c.log.Info("navigation succeeded - AI control re-enabled")

			c.mu.Lock()
			dest := c.destination
			c.mu.Unlock()
			if dest != "" {
				c.tts.AddText(fmt.Sprintf("Yaaay! I have reached the %s. Woof! Woof!", dest))
			} else {
				c.tts.AddText("Yaaay! I have reached my destination. Woof! Woof!")
			}
		}
	case statusCanceled, statusAborted:
		if c.navInProgress.Swap(false) {
			c.log.Warn("navigation ended without success - AI control remains disabled",
				zap.String("status", statusName(code)))
		}
	}
}

// publishAIStatus publishes an AIStatusRequest enabling or disabling AI control.
func (c *Connector) publishAIStatus(enabled bool) {
	if c.aiStatusPub == nil {
		return
	}

	code := byte(0)
	if enabled {
		code = 1
	}

	payload := serializeAIStatusRequest(frameMap, uuid.NewString(), code)
	if err := c.aiStatusPub.Put(payload); err != nil {
		c.log.Error("AI status put failed", zap.Error(err))
		return
	}
	c.log.Info("AI control request published", zap.Bool("enabled", enabled))
}

// Tick is a no-op; navigation status arrives via the Zenoh subscriber callback.
func (c *Connector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop releases the locations provider and all Zenoh resources.
func (c *Connector) Stop() {
	if c.locations != nil {
		c.locations.Stop()
	}
	if c.navStatusSub != nil {
		c.navStatusSub.Drop()
		c.navStatusSub = nil
	}
	if c.goalPosePub != nil {
		c.goalPosePub.Drop()
		c.goalPosePub = nil
	}
	if c.aiStatusPub != nil {
		c.aiStatusPub.Drop()
		c.aiStatusPub = nil
	}
	if c.session != nil {
		c.session.Close()
		c.session = nil
	}
	c.log.Info("connector stopped")
}

func cleanLabel(raw any) string {
	label, _ := raw.(string)
	label = util.TrimLower(label)
	for _, prefix := range labelPrefixes {
		if strings.HasPrefix(label, prefix) {
			return strings.TrimSpace(label[len(prefix):])
		}
	}
	return label
}
