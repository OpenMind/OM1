package llm

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

type fakeLLM struct {
	resp    *Response
	err     error
	schemas []map[string]any
	calls   []callRecord
}

type callRecord struct {
	prompt  string
	history []Message
}

func (f *fakeLLM) Call(_ context.Context, prompt string, history []Message) (*Response, error) {
	f.calls = append(f.calls, callRecord{prompt: prompt, history: append([]Message(nil), history...)})
	if f.err != nil {
		return nil, f.err
	}
	if f.resp != nil {
		return f.resp, nil
	}
	return &Response{}, nil
}

func (f *fakeLLM) SetSchemas(s []map[string]any)     { f.schemas = s }
func (f *fakeLLM) FunctionSchemas() []map[string]any { return f.schemas }

func TestRegisterAndLoad(t *testing.T) {
	want := &fakeLLM{}
	Register("FakeTestLLM", func(map[string]any) (LLM, error) { return want, nil })
	t.Cleanup(func() { delete(registry, "FakeTestLLM") })

	got, err := Load("FakeTestLLM", map[string]any{})
	require.NoError(t, err)
	require.Same(t, want, got)
}

func TestLoadUnknownPlugin(t *testing.T) {
	_, err := Load("DoesNotExist", nil)
	require.Error(t, err)
	var unknown *UnknownPluginError
	require.ErrorAs(t, err, &unknown)
	require.Equal(t, "DoesNotExist", unknown.Name)
	require.Equal(t, "llm plugin not found: DoesNotExist", err.Error())
}
