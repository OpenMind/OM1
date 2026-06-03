package actions

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

type stubConnector struct{}

func (stubConnector) Connect(context.Context, Input) (Output, error) { return nil, nil }
func (stubConnector) Tick(context.Context)                           {}
func (stubConnector) Stop()                                          {}

func TestActionLoadByCompositeKey(t *testing.T) {
	Register("greet/tts", func(map[string]any) (Connector, error) { return stubConnector{}, nil })
	t.Cleanup(func() { delete(connectorRegistry, "greet/tts") })

	action, err := Load("greet", "tts", "greetLabel", nil)
	require.NoError(t, err)
	require.Equal(t, "greet", action.Name)
	require.Equal(t, "greetLabel", action.LLMLabel)
	require.NotNil(t, action.Connector)
}

func TestActionLoadFallsBackToConnectorType(t *testing.T) {
	Register("speak", func(map[string]any) (Connector, error) { return stubConnector{}, nil })
	t.Cleanup(func() { delete(connectorRegistry, "speak") })

	action, err := Load("anything", "speak", "label", nil)
	require.NoError(t, err)
	require.NotNil(t, action.Connector)
}

func TestActionLoadUnknown(t *testing.T) {
	_, err := Load("missing", "nope", "label", nil)
	require.Error(t, err)
	var unknown *UnknownPluginError
	require.ErrorAs(t, err, &unknown)
	require.Equal(t, "action connector", unknown.Kind)
	require.Equal(t, "action connector plugin not found: missing/nope", err.Error())
}
