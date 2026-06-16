package providers

import (
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestConfidenceCalculatorDefaults(t *testing.T) {
	c := NewConfidenceCalculator()
	require.Equal(t, 5.0, c.SilenceThresholdSoft)
	require.Equal(t, 10.0, c.SilenceThresholdHard)
	require.InDelta(t, 1.0, c.weightLLM+c.weightSilence+c.weightEngagement+c.weightConversationQuality, 1e-9,
		"component weights sum to 1")
}

func TestLLMScoreByState(t *testing.T) {
	c := NewConfidenceCalculator()

	concluding := c.CalculateCompletionConfidence(ConfidenceFactors{
		ConversationState: StateConcluding, LLMConfidence: 0.8,
	})
	require.Equal(t, 0.8, concluding.Breakdown.LLM, "concluding uses confidence as-is")

	conversing := c.CalculateCompletionConfidence(ConfidenceFactors{
		ConversationState: StateConversing, LLMConfidence: 0.3,
	})
	require.InDelta(t, 0.7, conversing.Breakdown.LLM, 1e-9, "conversing inverts the confidence")

	idle := c.CalculateCompletionConfidence(ConfidenceFactors{ConversationState: StateIdle})
	require.Equal(t, 0.5, idle.Breakdown.LLM, "other states are neutral")
}

func TestSilenceScore(t *testing.T) {
	c := NewConfidenceCalculator()

	hard := c.CalculateCompletionConfidence(ConfidenceFactors{SilenceDuration: 12})
	require.Equal(t, 1.0, hard.Breakdown.Silence, "beyond hard threshold → 1.0")

	none := c.CalculateCompletionConfidence(ConfidenceFactors{SilenceDuration: 0})
	require.Equal(t, 0.0, none.Breakdown.Silence)

	mid := c.CalculateCompletionConfidence(ConfidenceFactors{SilenceDuration: 7.5})
	require.InDelta(t, 0.75, mid.Breakdown.Silence, 1e-9, "halfway between soft and hard → 0.75")
}

func TestEngagementScoreCapped(t *testing.T) {
	c := NewConfidenceCalculator()
	far := c.CalculateCompletionConfidence(ConfidenceFactors{PersonDistance: 100})
	require.Equal(t, 1.0, far.Breakdown.Engagement, "engagement score is capped at 1.0")
}

func TestShouldTransitionToConcluding(t *testing.T) {
	c := NewConfidenceCalculator()

	highOverall := ConfidenceResult{Overall: 0.8}
	require.True(t, c.ShouldTransitionToConcluding(highOverall), "high overall confidence concludes")

	llmStrong := ConfidenceResult{
		Overall:   0.55,
		Breakdown: ConfidenceBreakdown{LLM: 0.8},
		Factors:   ConfidenceFactors{ConversationState: StateConcluding},
	}
	require.True(t, c.ShouldTransitionToConcluding(llmStrong), "LLM wants to conclude with strong signal")

	mediumTwoSignals := ConfidenceResult{
		Overall:   0.65,
		Breakdown: ConfidenceBreakdown{LLM: 0.7, Silence: 0.7},
	}
	require.True(t, c.ShouldTransitionToConcluding(mediumTwoSignals), "two strong signals at medium confidence")

	weak := ConfidenceResult{Overall: 0.3}
	require.False(t, c.ShouldTransitionToConcluding(weak))
}

func TestShouldTransitionToFinished(t *testing.T) {
	c := NewConfidenceCalculator()

	require.True(t, c.ShouldTransitionToFinished(ConfidenceResult{Overall: 0.95}, 0), "very high confidence finishes immediately")
	require.True(t, c.ShouldTransitionToFinished(
		ConfidenceResult{Breakdown: ConfidenceBreakdown{Silence: 0.9}}, 4.0), "prolonged silence while concluding")
	require.True(t, c.ShouldTransitionToFinished(
		ConfidenceResult{Breakdown: ConfidenceBreakdown{Engagement: 0.9}}, 2.5), "person clearly leaving")
	require.True(t, c.ShouldTransitionToFinished(ConfidenceResult{Overall: 0.65}, 6.0), "timeout with reasonable confidence")
	require.False(t, c.ShouldTransitionToFinished(ConfidenceResult{Overall: 0.5}, 1.0))
}

func TestParseConversationState(t *testing.T) {
	log := zap.NewNop()
	require.Equal(t, StateConcluding, parseConversationState("concluding", log))
	require.Equal(t, StateFinished, parseConversationState("finished", log))
	require.Equal(t, StateConversing, parseConversationState("", log), "empty defaults to conversing")
	require.Equal(t, StateConversing, parseConversationState(42, log), "non-string defaults to conversing")
	require.Equal(t, StateConversing, parseConversationState("garbage", log), "unknown defaults to conversing")
}

func TestStateMachineLifecycle(t *testing.T) {
	g := NewGreetingConversationStateMachineProvider(3)
	require.Equal(t, StateIdle, g.CurrentState())
	require.Equal(t, 3, g.MaxTurnCount())

	g.StartConversation()
	require.Equal(t, StateConversing, g.CurrentState())
	require.Equal(t, 0, g.TurnCount())

	g.ResetState(StateIdle)
	require.Equal(t, StateConversing, g.CurrentState(), "ResetState ends in the conversing state")
	require.Equal(t, 0, g.TurnCount())
}

func TestDetermineNextStateForcesFinishAtMaxTurnCount(t *testing.T) {
	g := NewGreetingConversationStateMachineProvider(3)
	g.currentState = StateConversing

	r := ConfidenceResult{Factors: ConfidenceFactors{ConversationState: StateConversing}}

	g.turnCount = 2
	require.NotEqual(t, StateFinished, g.determineNextState(r),
		"turn below max keeps conversing")

	g.turnCount = 3
	require.Equal(t, StateFinished, g.determineNextState(r),
		"reaching max turn count forces finish")
}

func TestDetermineNextStateHonorsLLMFinished(t *testing.T) {
	g := NewGreetingConversationStateMachineProvider(3)
	g.currentState = StateConcluding
	g.turnCount = 1

	r := ConfidenceResult{Factors: ConfidenceFactors{ConversationState: StateFinished}}
	require.Equal(t, StateFinished, g.determineNextState(r),
		"an explicit finished state from the LLM ends the conversation")
}

func TestConfidenceHistoryTrend(t *testing.T) {
	g := NewGreetingConversationStateMachineProvider(3)
	require.Equal(t, "insufficient_data", g.confidenceTrend())

	for _, v := range []float64{0.1, 0.2, 0.5} {
		g.recordConfidence(v)
	}
	require.Equal(t, "increasing", g.confidenceTrend())

	g.confidenceHistory = nil
	for _, v := range []float64{0.8, 0.5, 0.2} {
		g.recordConfidence(v)
	}
	require.Equal(t, "decreasing", g.confidenceTrend())
}

func TestRecordConfidenceCapsHistory(t *testing.T) {
	g := NewGreetingConversationStateMachineProvider(3)
	for i := 0; i < 20; i++ {
		g.recordConfidence(float64(i))
	}
	require.Len(t, g.confidenceHistory, g.maxHistory, "history is bounded")
	require.Equal(t, 19.0, g.confidenceHistory[len(g.confidenceHistory)-1], "newest value is retained")
}
