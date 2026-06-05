package providers

import (
	"math"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/util"
)

// ConversationState enumerates the states of a greeting conversation.
type ConversationState string

const (
	StateIdle        ConversationState = "idle"
	StateApproaching ConversationState = "approaching"
	StateEngaging    ConversationState = "engaging"
	StateConversing  ConversationState = "conversing"
	StateConcluding  ConversationState = "concluding"
	StateFinished    ConversationState = "finished"
)

// ConfidenceFactors holds all factors that contribute to conversation completion
// confidence.
type ConfidenceFactors struct {
	ConversationState       ConversationState // from LLM output
	LLMConfidence           float64           // from LLM output
	SilenceDuration         float64           // seconds of silence
	SpeechClarity           float64           // audio quality / recognition confidence
	PersonDistance          float64           // meters from robot
	ConversationDuration    float64           // total conversation time
	TurnCount               int               // number of back-and-forth exchanges
	LastUserUtteranceLength int               // words in last user response
}

// ConfidenceBreakdown is the breakdown of individual confidence components.
type ConfidenceBreakdown struct {
	LLM        float64
	Silence    float64
	Engagement float64
	Quality    float64
}

// ConfidenceResult is the complete confidence calculation result.
type ConfidenceResult struct {
	Overall   float64
	Breakdown ConfidenceBreakdown
	Factors   ConfidenceFactors
}

// ConfidenceCalculator calculates the overall confidence score for conversation
// completion.
type ConfidenceCalculator struct {
	SilenceThresholdSoft    float64
	SilenceThresholdHard    float64
	MinConversationDuration float64
	EngagementDistanceMax   float64
	MinTurnCount            int

	weightLLM                 float64
	weightSilence             float64
	weightEngagement          float64
	weightConversationQuality float64
}

// NewConfidenceCalculator returns a ConfidenceCalculator with the default
// thresholds and weights.
func NewConfidenceCalculator() *ConfidenceCalculator {
	return &ConfidenceCalculator{
		SilenceThresholdSoft:    5.0,
		SilenceThresholdHard:    10.0,
		MinConversationDuration: 15.0,
		EngagementDistanceMax:   2.5,
		MinTurnCount:            2,

		weightLLM:                 0.35,
		weightSilence:             0.25,
		weightEngagement:          0.20,
		weightConversationQuality: 0.20,
	}
}

// CalculateCompletionConfidence computes the comprehensive confidence that a
// conversation should end.
func (c *ConfidenceCalculator) CalculateCompletionConfidence(factors ConfidenceFactors) ConfidenceResult {
	// LLM confidence
	var llmScore float64
	switch factors.ConversationState {
	case StateConcluding, StateFinished:
		// LLM wants to conclude - use confidence as-is
		llmScore = factors.LLMConfidence
	case StateConversing:
		// LLM wants to continue - invert the confidence for "end" score
		llmScore = math.Round((1.0-factors.LLMConfidence)*1000) / 1000
	default:
		// Other states (IDLE, APPROACHING, ENGAGING) - neutral
		llmScore = 0.5
	}

	// Silence factor
	var silenceScore float64
	switch {
	case factors.SilenceDuration >= c.SilenceThresholdHard:
		silenceScore = 1.0
	case factors.SilenceDuration >= c.SilenceThresholdSoft:
		silenceScore = 0.5 + 0.5*((factors.SilenceDuration-c.SilenceThresholdSoft)/
			(c.SilenceThresholdHard-c.SilenceThresholdSoft))
	default:
		silenceScore = factors.SilenceDuration / c.SilenceThresholdSoft * 0.5
	}

	// Engagement score (lower engagement = higher confidence to end)
	engagementScore := math.Min(factors.PersonDistance/c.EngagementDistanceMax, 1.0)

	// Conversation quality score (has it been substantial enough to end naturally?)
	durationScore := math.Min(factors.ConversationDuration/c.MinConversationDuration, 1.0)
	turnScore := math.Min(float64(factors.TurnCount)/float64(c.MinTurnCount), 1.0)

	// Short user utterances might indicate disengagement
	utteranceScore := 0.5
	if factors.LastUserUtteranceLength < 3 {
		utteranceScore = 1.0
	}

	qualityScore := (durationScore + turnScore + utteranceScore) / 3

	overall := c.weightLLM*llmScore +
		c.weightSilence*silenceScore +
		c.weightEngagement*engagementScore +
		c.weightConversationQuality*qualityScore

	return ConfidenceResult{
		Overall: overall,
		Breakdown: ConfidenceBreakdown{
			LLM:        llmScore,
			Silence:    silenceScore,
			Engagement: engagementScore,
			Quality:    qualityScore,
		},
		Factors: factors,
	}
}

// ShouldTransitionToConcluding decides if we should move to the concluding state.
func (c *ConfidenceCalculator) ShouldTransitionToConcluding(r ConfidenceResult) bool {
	overall := r.Overall
	b := r.Breakdown
	state := r.Factors.ConversationState

	llmWantsToConclude := state == StateConcluding || state == StateFinished

	if llmWantsToConclude && b.LLM >= 0.7 {
		if overall >= 0.5 {
			return true
		}
	}

	// High overall confidence
	if overall >= 0.75 {
		return true
	}

	// Medium confidence but strong signals from multiple sources
	if overall >= 0.6 {
		strongSignals := 0
		if b.LLM >= 0.7 {
			strongSignals++
		}
		if b.Silence >= 0.6 {
			strongSignals++
		}
		if b.Engagement >= 0.6 {
			strongSignals++
		}
		if strongSignals >= 2 {
			return true
		}
	}

	return false
}

// ShouldTransitionToFinished decides if we should move to the finished state.
func (c *ConfidenceCalculator) ShouldTransitionToFinished(r ConfidenceResult, timeInConcluding float64) bool {
	overall := r.Overall
	b := r.Breakdown

	// Very high confidence - immediate transition
	if overall >= 0.9 {
		return true
	}

	// High silence after we've been concluding
	if timeInConcluding >= 3.0 && b.Silence >= 0.8 {
		return true
	}

	// Person is clearly leaving
	if b.Engagement >= 0.8 && timeInConcluding >= 2.0 {
		return true
	}

	// Timeout with reasonable confidence
	if timeInConcluding >= 5.0 && overall >= 0.6 {
		return true
	}

	return false
}

// GreetingConversationStateMachineProvider manages greeting conversation state
// transitions based on confidence scores.
type GreetingConversationStateMachineProvider struct {
	mu sync.Mutex

	currentState          ConversationState
	previousState         ConversationState
	stateEntryTime        time.Time
	conversationStartTime time.Time // zero value means "not started"
	turnCount             int
	maxTurnCount          int
	lastUserUtterance     string

	calc *ConfidenceCalculator

	confidenceHistory []float64
	maxHistory        int

	io  *IOProvider
	log *zap.Logger
}

var (
	greetingOnce     sync.Once
	greetingInstance *GreetingConversationStateMachineProvider
)

// Greeting returns the singleton GreetingConversationStateMachineProvider.
func Greeting() *GreetingConversationStateMachineProvider {
	greetingOnce.Do(func() { greetingInstance = NewGreetingConversationStateMachineProvider(3) })
	return greetingInstance
}

// NewGreetingConversationStateMachineProvider constructs a state machine with the
// given maximum turn count before forcing a transition to finished.
func NewGreetingConversationStateMachineProvider(maxTurnCount int) *GreetingConversationStateMachineProvider {
	return &GreetingConversationStateMachineProvider{
		currentState:   StateIdle,
		stateEntryTime: time.Now(),
		maxTurnCount:   maxTurnCount,
		calc:           NewConfidenceCalculator(),
		maxHistory:     5,
		io:             IO(),
		log:            logger.Get(),
	}
}

// ProcessConversation processes greeting conversation state based on LLM output
// and other factors, returning the current state and confidence details.
func (g *GreetingConversationStateMachineProvider) ProcessConversation(llmOutput map[string]any) map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()

	// TODO: integrate person distance, speech clarity, etc.
	personDistance := 1.5 // placeholder

	voice := g.io.GetInput("Voice")
	silenceDuration := time.Since(g.stateEntryTime).Seconds()
	if voice != nil && !voice.Timestamp.IsZero() {
		silenceDuration = time.Since(voice.Timestamp).Seconds()
	}

	conversationState := parseConversationState(llmOutput["conversation_state"], g.log)

	factors := ConfidenceFactors{
		ConversationState:       conversationState,
		LLMConfidence:           util.FloatFrom(llmOutput["confidence"], 0.0),
		SilenceDuration:         silenceDuration,
		SpeechClarity:           util.FloatFrom(llmOutput["speech_clarity"], 1.0),
		PersonDistance:          personDistance,
		ConversationDuration:    g.conversationDuration(),
		TurnCount:               g.turnCount,
		LastUserUtteranceLength: len([]rune(g.lastUserUtterance)),
	}

	result := g.calc.CalculateCompletionConfidence(factors)
	g.recordConfidence(result.Overall)

	newState := g.determineNextState(result)
	g.previousState = g.currentState
	if newState != g.currentState {
		g.currentState = newState
		g.stateEntryTime = time.Now()
		g.log.Info("greeting: state transition",
			zap.String("from", string(g.previousState)),
			zap.String("to", string(g.currentState)),
			zap.Float64("confidence", result.Overall),
		)
	}

	// Count a turn when the latest voice input arrived during the current tick.
	if voice != nil && g.io.TickCounter() == voice.Tick {
		g.turnCount++
		g.lastUserUtterance = voice.Input
	}

	command := g.generateCommand(result)

	return map[string]any{
		"current_state":    string(g.currentState),
		"confidence":       result,
		"command":          command,
		"time_in_state":    time.Since(g.stateEntryTime).Seconds(),
		"confidence_trend": g.confidenceTrend(),
		"turn_count":       g.turnCount,
	}
}

// UpdateStateWithoutLLM updates conversation state based on current factors
// without requiring LLM input. It should be called periodically to handle
// timeouts and silence-driven transitions.
func (g *GreetingConversationStateMachineProvider) UpdateStateWithoutLLM() map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()

	personDistance := 1.5 // placeholder

	silenceDuration := time.Since(g.stateEntryTime).Seconds()
	if voice := g.io.GetInput("Voice"); voice != nil && !voice.Timestamp.IsZero() {
		silenceDuration = time.Since(voice.Timestamp).Seconds()
	}

	factors := ConfidenceFactors{
		ConversationState:       g.currentState,
		LLMConfidence:           0.5, // neutral confidence when no LLM input
		SilenceDuration:         silenceDuration,
		SpeechClarity:           1.0,
		PersonDistance:          personDistance,
		ConversationDuration:    g.conversationDuration(),
		TurnCount:               g.turnCount,
		LastUserUtteranceLength: len([]rune(g.lastUserUtterance)),
	}

	result := g.calc.CalculateCompletionConfidence(factors)
	g.recordConfidence(result.Overall)

	newState := g.determineNextState(result)
	g.previousState = g.currentState
	if newState != g.currentState {
		g.log.Info("greeting: state transition",
			zap.String("from", string(g.currentState)),
			zap.String("to", string(newState)),
		)
		g.currentState = newState
		g.stateEntryTime = time.Now()
	}

	command := g.generateCommand(result)

	return map[string]any{
		"current_state":    string(g.currentState),
		"confidence":       result,
		"command":          command,
		"time_in_state":    time.Since(g.stateEntryTime).Seconds(),
		"confidence_trend": g.confidenceTrend(),
		"silence_duration": silenceDuration,
	}
}

// determineNextState determines the next conversation state based on LLM output
// and confidence. Callers must hold g.mu.
func (g *GreetingConversationStateMachineProvider) determineNextState(r ConfidenceResult) ConversationState {
	llmState := r.Factors.ConversationState
	timeInState := time.Since(g.stateEntryTime).Seconds()

	// Force finish if we've had too many turns to prevent infinite conversations
	if g.turnCount >= g.maxTurnCount {
		g.log.Info("greeting: maximum turn count reached - forcing finish",
			zap.Int("max_turn_count", g.maxTurnCount))
		return StateFinished
	}

	switch g.currentState {
	case StateConversing:
		// LLM thinks we should conclude with high confidence
		if llmState == StateConcluding && r.Breakdown.LLM >= 0.7 {
			g.log.Info("greeting: LLM indicates concluding with high confidence")
			return StateConcluding
		}
		// Traditional confidence-based transition
		if g.calc.ShouldTransitionToConcluding(r) {
			g.log.Info("greeting: transitioning to CONCLUDING based on confidence")
			return StateConcluding
		}

	case StateConcluding:
		if g.calc.ShouldTransitionToFinished(r, timeInState) {
			g.log.Info("greeting: transitioning to FINISHED based on confidence")
			return StateFinished
		}
		// Revert to CONVERSING if the LLM says we're still conversing
		if llmState == StateConversing {
			g.log.Info("greeting: LLM indicates conversation continuing, reverting to conversing")
			return StateConversing
		}
		if r.Overall < 0.4 && r.Breakdown.Engagement < 0.3 {
			g.log.Info("greeting: low confidence and strong engagement, reverting to conversing")
			return StateConversing
		}
	}

	// Emergency timeout - force finish after very long concluding
	if g.currentState == StateConcluding && timeInState > 15.0 {
		g.log.Info("greeting: emergency timeout - forcing finish")
		return StateFinished
	}

	return g.currentState
}

// confidenceTrend analyzes whether confidence is increasing or decreasing.
// Callers must hold g.mu.
func (g *GreetingConversationStateMachineProvider) confidenceTrend() string {
	if len(g.confidenceHistory) < 3 {
		return "insufficient_data"
	}
	recent := g.confidenceHistory[len(g.confidenceHistory)-3:]
	switch {
	case recent[len(recent)-1] > recent[0]+0.1:
		return "increasing"
	case recent[len(recent)-1] < recent[0]-0.1:
		return "decreasing"
	default:
		return "stable"
	}
}

// generateCommand generates a command with confidence metadata. Callers must
// hold g.mu.
func (g *GreetingConversationStateMachineProvider) generateCommand(r ConfidenceResult) map[string]any {
	cmd := g.baseCommand()
	cmd["confidence_metadata"] = map[string]any{
		"overall":   r.Overall,
		"breakdown": r.Breakdown,
		"trend":     g.confidenceTrend(),
	}
	return cmd
}

// baseCommand returns the base robot command for the current conversation state.
// Callers must hold g.mu.
func (g *GreetingConversationStateMachineProvider) baseCommand() map[string]any {
	switch g.currentState {
	case StateFinished:
		return map[string]any{
			"action":          "move_to_next_person",
			"track_current":   false,
			"find_new_target": true,
		}
	case StateConcluding:
		return map[string]any{
			"action":          "maintain_position",
			"track_current":   true,
			"find_new_target": false,
		}
	case StateConversing:
		return map[string]any{
			"action":          "stay_with_person",
			"track_current":   true,
			"find_new_target": false,
		}
	default:
		return map[string]any{
			"action":          "wait",
			"track_current":   true,
			"find_new_target": false,
		}
	}
}

// conversationDuration returns the total conversation duration in seconds, or 0
// if no conversation has started. Callers must hold g.mu.
func (g *GreetingConversationStateMachineProvider) conversationDuration() float64 {
	if g.conversationStartTime.IsZero() {
		return 0.0
	}
	return time.Since(g.conversationStartTime).Seconds()
}

// recordConfidence appends an overall confidence score, capping history length.
// Callers must hold g.mu.
func (g *GreetingConversationStateMachineProvider) recordConfidence(overall float64) {
	g.confidenceHistory = append(g.confidenceHistory, overall)
	if len(g.confidenceHistory) > g.maxHistory {
		g.confidenceHistory = g.confidenceHistory[len(g.confidenceHistory)-g.maxHistory:]
	}
}

// StartConversation should be called when starting a new conversation.
func (g *GreetingConversationStateMachineProvider) StartConversation() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.conversationStartTime = time.Now()
	g.turnCount = 0
	g.confidenceHistory = nil
	g.currentState = StateConversing
}

// ResetState resets the conversation state machine to initial values. It is
// typically called when transitioning from approaching to greeting.
func (g *GreetingConversationStateMachineProvider) ResetState(initialState ConversationState) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.currentState = initialState
	g.previousState = ""
	g.stateEntryTime = time.Now()
	g.conversationStartTime = time.Now()
	g.turnCount = 0
	g.lastUserUtterance = ""
	g.confidenceHistory = nil
	g.currentState = StateConversing
}

// CurrentState returns the current conversation state.
func (g *GreetingConversationStateMachineProvider) CurrentState() ConversationState {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.currentState
}

// TurnCount returns the number of back-and-forth exchanges in the current
// conversation.
func (g *GreetingConversationStateMachineProvider) TurnCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.turnCount
}

// MaxTurnCount returns the maximum turn count before the conversation is forced
// to finish.
func (g *GreetingConversationStateMachineProvider) MaxTurnCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.maxTurnCount
}

// finalTurnGuidance is the guidance string for the LLM when we're in the concluding state or approaching the max turn count.
const finalTurnGuidance = "\nGreeting status: This is the final exchange of this greeting conversation. " +
	"Give a brief, warm goodbye in your response and set conversation_state to \"finished\".\n"

// EndingGuidance returns a guidance string for the LLM when we're in the concluding state or approaching the max turn count.
func (g *GreetingConversationStateMachineProvider) EndingGuidance() string {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.currentState == StateFinished {
		return ""
	}
	if g.currentState == StateConcluding || g.turnCount+1 >= g.maxTurnCount {
		return finalTurnGuidance
	}
	return ""
}

// parseConversationState coerces an LLM-provided value into a ConversationState,
// defaulting to CONVERSING on unknown values.
func parseConversationState(raw any, log *zap.Logger) ConversationState {
	s, ok := raw.(string)
	if !ok || s == "" {
		return StateConversing
	}
	switch ConversationState(s) {
	case StateIdle, StateApproaching, StateEngaging, StateConversing, StateConcluding, StateFinished:
		return ConversationState(s)
	default:
		log.Warn("greeting: invalid conversation_state from LLM, defaulting to CONVERSING",
			zap.String("conversation_state", s))
		return StateConversing
	}
}
