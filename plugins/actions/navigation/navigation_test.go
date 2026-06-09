package navigation

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/providers/tts"
)

func TestParseConfigDefaults(t *testing.T) {
	cfg := parseConfig(nil)

	assert.Equal(t, localLocationsURL, cfg.BaseURL, "no use_sim → local locations URL")
	assert.Equal(t, defaultGoalPoseTopic, cfg.GoalPoseTopic)
	assert.Equal(t, defaultNavStatusTopic, cfg.NavStatusTopic)
	assert.Equal(t, defaultAIStatusTopic, cfg.AIStatusTopic)
	assert.Equal(t, tts.DefaultVoiceID, cfg.VoiceID)
	assert.Equal(t, tts.DefaultRate, cfg.Rate)
}

func TestParseConfigUseSimBaseURL(t *testing.T) {
	cfg := parseConfig(map[string]any{"use_sim": true})
	assert.Equal(t, simLocationsURL, cfg.BaseURL, "use_sim with no base_url → simulation URL")
}

func TestParseConfigExplicitValuesPreserved(t *testing.T) {
	cfg := parseConfig(map[string]any{
		"base_url":         "http://example.com/list",
		"use_sim":          true,
		"goal_pose_topic":  "custom_goal",
		"refresh_interval": 10,
	})
	assert.Equal(t, "http://example.com/list", cfg.BaseURL, "explicit base_url overrides use_sim default")
	assert.Equal(t, "custom_goal", cfg.GoalPoseTopic)
	assert.Equal(t, 10, cfg.RefreshInterval)
}

func TestCleanLabel(t *testing.T) {
	cases := map[string]string{
		"kitchen":               "kitchen",
		"Kitchen":               "kitchen",
		"  sofa  ":              "sofa",
		"go to the kitchen":     "kitchen",
		"GO TO sofa":            "sofa",
		"navigate to the table": "table",
		"move to garage":        "garage",
		"take me to the lobby":  "lobby",
		"":                      "",
	}
	for input, want := range cases {
		assert.Equalf(t, want, cleanLabel(input), "cleanLabel(%q)", input)
	}
}

func TestCleanLabelNonString(t *testing.T) {
	assert.Equal(t, "", cleanLabel(nil))
	assert.Equal(t, "", cleanLabel(42))
}

func newTestConnector() *Connector {
	return &Connector{
		log: zap.NewNop(),
		tts: tts.ElevenLabs(tts.ElevenLabsConfig{
			OutputFormat: tts.DefaultOutputFormat,
			Rate:         tts.DefaultRate,
		}, zap.NewNop()),
	}
}

func TestConnectWrongInputType(t *testing.T) {
	c := newTestConnector()

	out, err := c.Connect(context.Background(), "not-a-map")

	require.Error(t, err)
	assert.Nil(t, out)
}

func TestConnectorImplementsInterface(t *testing.T) {
	var _ actions.Connector = (*Connector)(nil)
}
