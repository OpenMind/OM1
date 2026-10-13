package runtime

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/llm"
)

func TestRepeatGuardSuppressesIdenticalRepeat(t *testing.T) {
	g := &repeatGuard{}
	wave := []llm.ToolCall{{Name: "robot_action", Arguments: map[string]any{"action": "wave"}}}

	require.True(t, g.Allow("classifier confirms change", wave), "first response to an observation runs")
	require.False(t, g.Allow("classifier confirms change", wave), "the same response to the same observation is dropped")
	require.True(t, g.Allow("classifier confirms change again", wave), "a new observation resets the guard")
	require.False(t, g.Allow("classifier confirms change again", wave))
}

func TestRepeatGuardIdleTicks(t *testing.T) {
	g := &repeatGuard{}
	wave := []llm.ToolCall{{Name: "robot_action", Arguments: map[string]any{"action": "wave"}}}
	look := []llm.ToolCall{{Name: "robot_action", Arguments: map[string]any{"action": "look"}}}

	require.True(t, g.Allow("", wave), "an action on an idle tick runs once")
	require.False(t, g.Allow("", wave), "and is not repeated on the next idle tick")
	require.True(t, g.Allow("", look), "a different action on an idle tick runs")
	require.True(t, g.Allow("", nil), "doing nothing is always allowed")
	require.True(t, g.Allow("", look), "after a no-op the guard has reset")
}

func TestRepeatGuardOrderAndMultipleActions(t *testing.T) {
	g := &repeatGuard{}
	a := llm.ToolCall{Name: "speak", Arguments: map[string]any{"sentence": "hi"}}
	b := llm.ToolCall{Name: "emotion", Arguments: map[string]any{"emotion": "happy"}}

	require.True(t, g.Allow("x", []llm.ToolCall{a, b}))
	require.False(t, g.Allow("x", []llm.ToolCall{b, a}), "order does not matter")
	require.True(t, g.Allow("x", []llm.ToolCall{a}), "a different set runs")
}

func TestActionKey(t *testing.T) {
	require.Equal(t, "", actionKey(nil))
	k1 := actionKey([]llm.ToolCall{{Name: "speak", Arguments: map[string]any{"sentence": "hi"}}})
	k2 := actionKey([]llm.ToolCall{{Name: "speak", Arguments: map[string]any{"sentence": "hi"}}})
	k3 := actionKey([]llm.ToolCall{{Name: "speak", Arguments: map[string]any{"sentence": "bye"}}})
	require.Equal(t, k1, k2)
	require.NotEqual(t, k1, k3)
}

func TestFilterSpeech(t *testing.T) {
	speak := llm.ToolCall{Name: "speak", Arguments: map[string]any{"sentence": "hello"}}
	move := llm.ToolCall{Name: "move", Arguments: map[string]any{"direction": "forward"}}

	kept, dropped := filterSpeech([]llm.ToolCall{speak, move}, []string{"speak"}, false)
	require.Equal(t, 1, dropped)
	require.Equal(t, []llm.ToolCall{move}, kept, "speech is dropped, other actions run")

	kept, dropped = filterSpeech([]llm.ToolCall{speak, move}, []string{"speak"}, true)
	require.Equal(t, 0, dropped)
	require.Len(t, kept, 2, "a conversation turn or event allows speech")

	kept, dropped = filterSpeech([]llm.ToolCall{speak}, nil, false)
	require.Equal(t, 0, dropped)
	require.Len(t, kept, 1, "no speech list means no gate")

	kept, dropped = filterSpeech(nil, []string{"speak"}, false)
	require.Equal(t, 0, dropped)
	require.Empty(t, kept)
}
