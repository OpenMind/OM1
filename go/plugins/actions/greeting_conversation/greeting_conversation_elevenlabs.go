package greeting_conversation

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
	"github.com/openmind/om1/internal/zenoh"
)

const (
	// personGreetingTopic carries countdown / conversation status updates.
	personGreetingTopic = "om/person_greeting"
	// contextUpdateTopic carries user-context updates that drive context-aware
	// mode transitions (the ModeManager subscribes here).
	contextUpdateTopic = "om/mode/context"

	// statusConversation is PersonGreetingStatus.STATUS.CONVERSATION.
	statusConversation = byte(3)

	// tickInterval matches the Python connector's per-tick sleep.
	tickInterval = 10 * time.Second
	// ttsTimeout caps how long we wait for TTS playback before resuming ticks.
	ttsTimeout = 30 * time.Second
)

// ConversationStateEnum is the LLM-facing enum for the conversation_state field.
type ConversationStateEnum string

// EnumValues lists the conversation states the LLM may report.
func (ConversationStateEnum) EnumValues() []string {
	return []string{
		string(providers.StateConversing),
		string(providers.StateConcluding),
		string(providers.StateFinished),
	}
}

// GreetingConversationInput is the input the LLM provides for a greeting turn.
type GreetingConversationInput struct {
	Response          string                `json:"response" description:"The spoken text response to the user"`
	ConversationState ConversationStateEnum `json:"conversation_state" description:"The current state of the greeting conversation"`
	Confidence        float64               `json:"confidence" description:"Confidence score of the conversation state recognition from 0.0 to 1.0"`
	SpeechClarity     float64               `json:"speech_clarity" description:"The user's voice input clarity score from 0.0 to 1.0"`
}

func init() {
	actions.RegisterInterface(
		"greeting_conversation",
		"Action interface for greeting conversations. Speaks a response to the user, "+
			"tracks the conversation state (conversing, concluding, finished), and reports "+
			"recognition confidence and the user's speech clarity.",
		GreetingConversationInput{},
	)
	actions.Register("greeting_conversation/greeting_conversation_elevenlabs", NewElevenLabsGreetingConversation)
}

// Config holds the ElevenLabs TTS parameters for the greeting connector.
type Config struct {
	APIKey           string `json:"api_key"`
	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
	ZenohEndpoint    string `json:"zenoh_endpoint"`
}

// Connector manages a greeting conversation using ElevenLabs TTS. It is a Go
// port of the Python BaseGreetingConversationConnector + ElevenLabs subclass.
type Connector struct {
	log      *zap.Logger
	tts      *providers.ElevenLabsProvider
	greeting *providers.GreetingConversationStateMachineProvider
	session  *zenoh.Session

	mu                       sync.Mutex
	greetingStatus           providers.ConversationState
	conversationFinishedSent bool
	pendingFinishedUpdate    bool
	ttsPlaying               bool
	ttsPlayingStart          time.Time
}

// NewElevenLabsGreetingConversation builds the connector from a decoded config map.
func NewElevenLabsGreetingConversation(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("greeting_conversation/greeting_conversation_elevenlabs: api_key required")
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = providers.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = providers.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = providers.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = providers.DefaultRate
	}

	log := logger.Get()

	tts := providers.ElevenLabs(providers.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	greeting := providers.Greeting()
	greeting.StartConversation()

	c := &Connector{
		log:            log,
		tts:            tts,
		greeting:       greeting,
		greetingStatus: providers.StateConversing,
	}

	if sess, err := zenoh.Open(cfg.ZenohEndpoint); err != nil {
		log.Warn("greeting_conversation: Zenoh unavailable, status/context updates disabled", zap.Error(err))
	} else {
		c.session = sess
		log.Info("greeting_conversation: Zenoh session opened")
	}

	return c, nil
}

// Connect processes one greeting turn: speak the response, advance the
// conversation state machine, and publish the resulting status.
func (c *Connector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("greeting_conversation: unexpected input type %T", input)
	}

	response, _ := args["response"].(string)
	conversationState, _ := args["conversation_state"].(string)
	confidence := util.FloatFrom(args["confidence"], 0.0)
	speechClarity := util.FloatFrom(args["speech_clarity"], 1.0)

	c.log.Info("greeting_conversation: turn",
		zap.String("conversation_state", conversationState),
		zap.String("response", response),
		zap.Float64("confidence", confidence),
		zap.Float64("speech_clarity", speechClarity),
	)

	llmOutput := map[string]any{
		"conversation_state": conversationState,
		"response":           response,
		"confidence":         confidence,
		"speech_clarity":     speechClarity,
	}

	if response != "" {
		c.tts.AddText(normalizeTTSText(response))
		c.mu.Lock()
		c.ttsPlaying = true
		c.ttsPlayingStart = time.Now()
		c.mu.Unlock()
	}

	stateUpdate := c.greeting.ProcessConversation(llmOutput)
	currentState := stateString(stateUpdate, c.currentStatus())

	c.mu.Lock()
	c.greetingStatus = currentState
	c.mu.Unlock()
	c.publishCountdown(currentState)

	c.log.Info("greeting_conversation: state update", zap.Any("state_update", stateUpdate))

	if currentState == providers.StateFinished {
		c.handleFinished()
	}

	return nil, nil
}

// Tick periodically advances the conversation state without LLM input, handling
// silence-driven transitions and timeouts.
func (c *Connector) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
		return
	case <-time.After(tickInterval):
	}

	if c.waitingOnTTS() {
		c.log.Info("greeting_conversation: skipping tick during active TTS playback")
		return
	}

	stateUpdate := c.greeting.UpdateStateWithoutLLM()
	currentState := stateString(stateUpdate, c.currentStatus())

	c.mu.Lock()
	c.greetingStatus = currentState
	c.mu.Unlock()
	c.publishCountdown(currentState)

	if currentState == providers.StateFinished {
		c.mu.Lock()
		send := !c.conversationFinishedSent
		c.conversationFinishedSent = true
		c.mu.Unlock()
		if send {
			c.log.Info("greeting_conversation: finished (detected in tick)")
			c.updateContextFinished()
		}
	}
}

// Stop releases the Zenoh session. The shared ElevenLabs provider manages its
// own lifecycle and is not stopped here.
func (c *Connector) Stop() {
	c.log.Info("greeting_conversation: stopping")
	c.mu.Lock()
	sess := c.session
	c.session = nil
	c.mu.Unlock()
	if sess != nil {
		sess.Close()
		c.log.Info("greeting_conversation: Zenoh session closed")
	}
}

// handleFinished records the finished transition and updates context, deferring
// the context update until TTS playback completes if it is still in progress.
func (c *Connector) handleFinished() {
	c.mu.Lock()
	if c.conversationFinishedSent {
		c.mu.Unlock()
		return
	}
	c.conversationFinishedSent = true
	speaking := c.ttsPlaying
	if speaking {
		c.pendingFinishedUpdate = true
	}
	c.mu.Unlock()

	if speaking {
		c.log.Info("greeting_conversation: finished, waiting for TTS to complete before updating context")
		return
	}
	c.log.Info("greeting_conversation: finished")
	c.updateContextFinished()
}

// waitingOnTTS reports whether playback is still active. When playback has just
// completed it flushes any pending finished-context update; on timeout it gives
// up waiting so ticks can resume.
func (c *Connector) waitingOnTTS() bool {
	c.mu.Lock()
	if !c.ttsPlaying {
		c.mu.Unlock()
		return false
	}
	elapsed := time.Since(c.ttsPlayingStart)
	stillSpeaking := providers.Speaking.Load()

	if stillSpeaking && elapsed <= ttsTimeout {
		c.mu.Unlock()
		return true
	}

	c.ttsPlaying = false
	timedOut := stillSpeaking && elapsed > ttsTimeout
	pending := c.pendingFinishedUpdate
	c.pendingFinishedUpdate = false
	c.mu.Unlock()

	if timedOut {
		c.log.Warn("greeting_conversation: TTS playback timed out", zap.Duration("elapsed", elapsed))
	}
	if pending {
		c.log.Info("greeting_conversation: TTS completed, updating context greeting_conversation_finished=true")
		c.updateContextFinished()
	}
	return false
}

// publishCountdown publishes a PersonGreetingStatus carrying the seconds
// remaining until the conversation is expected to finish.
func (c *Connector) publishCountdown(state providers.ConversationState) {
	var secondsUntilFinished int
	switch state {
	case providers.StateConversing:
		secondsUntilFinished = 20
	case providers.StateConcluding:
		secondsUntilFinished = 10
	default:
		secondsUntilFinished = 0
	}

	c.mu.Lock()
	sess := c.session
	c.mu.Unlock()
	if sess == nil {
		return
	}

	requestID := uuid.New().String()
	message, _ := json.Marshal(map[string]any{"seconds_until_finished": secondsUntilFinished})
	payload := serializePersonGreetingStatus(requestID, statusConversation, string(message))

	if err := sess.Put(personGreetingTopic, payload); err != nil {
		c.log.Error("greeting_conversation: failed to publish PersonGreetingStatus", zap.Error(err))
		return
	}
	c.log.Info("greeting_conversation: published PersonGreetingStatus", zap.ByteString("message", message))
}

// updateContextFinished publishes greeting_conversation_finished=true so the
// ModeManager can fire its context-aware transition out of greeting mode.
func (c *Connector) updateContextFinished() {
	c.mu.Lock()
	sess := c.session
	c.mu.Unlock()
	if sess == nil {
		c.log.Warn("greeting_conversation: no Zenoh session, cannot update context")
		return
	}

	payload, _ := json.Marshal(map[string]any{"greeting_conversation_finished": true})
	if err := sess.Put(contextUpdateTopic, payload); err != nil {
		c.log.Error("greeting_conversation: failed to publish context update", zap.Error(err))
		return
	}
	c.log.Info("greeting_conversation: context greeting_conversation_finished=true")
}

func (c *Connector) currentStatus() providers.ConversationState {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.greetingStatus
}

// stateString extracts current_state from a state-machine result, falling back
// to the previous status when absent.
func stateString(stateUpdate map[string]any, fallback providers.ConversationState) providers.ConversationState {
	if s, ok := stateUpdate["current_state"].(string); ok && s != "" {
		return providers.ConversationState(s)
	}
	return fallback
}
