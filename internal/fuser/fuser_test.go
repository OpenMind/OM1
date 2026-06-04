package fuser

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/providers"
)

type fakeKB struct {
	docs []string
	err  error
	gotQ string
}

func (k *fakeKB) Query(_ context.Context, question string, _ int) ([]string, error) {
	k.gotQ = question
	return k.docs, k.err
}

func newRC() *config.RuntimeConfig {
	return &config.RuntimeConfig{SystemPromptBase: "You are a robot."}
}

func TestFuseBasic(t *testing.T) {
	f := NewFuser(newRC(), nil, nil, zap.NewNop())
	out, err := f.Fuse(context.Background(), nil)
	require.NoError(t, err)
	require.Contains(t, out, "You are a robot.")
	require.Contains(t, out, "What will you do next?")
	require.NotContains(t, out, "Current observations:")
}

func TestFuseIncludesGovernanceAndExamples(t *testing.T) {
	rc := newRC()
	rc.SystemGovernance = "Be safe."
	rc.PromptExamples = "Example: greet."
	out, err := NewFuser(rc, nil, nil, zap.NewNop()).Fuse(context.Background(), nil)
	require.NoError(t, err)
	require.Contains(t, out, "Governance rules:")
	require.Contains(t, out, "Be safe.")
	require.Contains(t, out, "Example: greet.")
}

func TestFuseIncludesObservations(t *testing.T) {
	out, err := NewFuser(newRC(), nil, nil, zap.NewNop()).
		Fuse(context.Background(), []string{"saw a person", "", "heard a sound"})
	require.NoError(t, err)
	require.Contains(t, out, "Current observations:")
	require.Contains(t, out, "- saw a person")
	require.Contains(t, out, "- heard a sound")
	require.NotContains(t, out, "- \n", "empty buffers are skipped")
}

func TestFuseVisibleActions(t *testing.T) {
	acts := []*actions.AgentAction{
		{LLMLabel: "speak"},
		{LLMLabel: "hidden", ExcludeFromPrompt: true},
		{LLMLabel: "move"},
	}
	out, err := NewFuser(newRC(), acts, nil, zap.NewNop()).Fuse(context.Background(), nil)
	require.NoError(t, err)
	require.Contains(t, out, "Available actions:")
	require.Contains(t, out, "- speak")
	require.Contains(t, out, "- move")
	require.NotContains(t, out, "hidden", "ExcludeFromPrompt actions are omitted")
}

func TestFuseIncludesKnowledgeBase(t *testing.T) {
	// A live voice input on the current tick drives the KB query.
	io := providers.IO()
	io.ResetTickCounter()
	io.IncrementTick()
	io.AddInput("Voice", "what time is it?", time.Now())
	t.Cleanup(func() { io.RemoveInput("Voice"); io.ResetTickCounter() })

	rc := newRC()
	rc.KnowledgeBase = &config.KBSpec{Name: "kb", TopK: 2}
	kb := &fakeKB{docs: []string{"It is noon.", "The museum opens at 9."}}

	out, err := NewFuser(rc, nil, kb, zap.NewNop()).Fuse(context.Background(), nil)
	require.NoError(t, err)
	require.Equal(t, "what time is it?", kb.gotQ, "the voice query is forwarded to the KB")
	require.Contains(t, out, "Relevant context:")
	require.Contains(t, out, "- It is noon.")
}

func TestFuseSkipsKBWhenNoVoice(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.RemoveInput("Voice")

	rc := newRC()
	rc.KnowledgeBase = &config.KBSpec{Name: "kb"}
	kb := &fakeKB{docs: []string{"unused"}}

	out, err := NewFuser(rc, nil, kb, zap.NewNop()).Fuse(context.Background(), nil)
	require.NoError(t, err)
	require.Empty(t, kb.gotQ, "KB is not queried without a fresh voice input")
	require.NotContains(t, out, "Relevant context:")
}

func TestFuseToleratesKBError(t *testing.T) {
	io := providers.IO()
	io.ResetTickCounter()
	io.IncrementTick()
	io.AddInput("Voice", "hello", time.Now())
	t.Cleanup(func() { io.RemoveInput("Voice"); io.ResetTickCounter() })

	rc := newRC()
	rc.KnowledgeBase = &config.KBSpec{Name: "kb"}
	kb := &fakeKB{err: context.DeadlineExceeded}

	out, err := NewFuser(rc, nil, kb, zap.NewNop()).Fuse(context.Background(), nil)
	require.NoError(t, err, "a KB failure is logged, not fatal")
	require.NotContains(t, out, "Relevant context:")
}
