package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestExpandEnv(t *testing.T) {
	t.Setenv("OM1_TEST_VAR", "world")
	require.Equal(t, "hello world", expandEnv("hello ${OM1_TEST_VAR}"))
	require.Equal(t, "hello fallback", expandEnv("hello ${OM1_MISSING_VAR:-fallback}"), "uses default when unset")
	require.Equal(t, "hello ", expandEnv("hello ${OM1_MISSING_VAR}"), "empty when unset and no default")
}

func TestCommentIndex(t *testing.T) {
	require.Equal(t, 4, commentIndex(`abc // comment`))
	require.Equal(t, -1, commentIndex(`"http://example.com"`), "// inside a string is ignored")
	require.Equal(t, -1, commentIndex(`no comment here`))
}

func TestStripJSON5(t *testing.T) {
	src := []byte(`{
		// a comment
		key: "value", // trailing
		nested: {
			url: "http://x/y", // url with slashes
		},
		list: [1, 2,],
	}`)
	out, err := stripJSON5(src)
	require.NoError(t, err)
	require.Contains(t, string(out), `"key": "value"`, "unquoted keys are quoted")
	require.Contains(t, string(out), `"url": "http://x/y"`, "urls inside strings survive")
	require.NotContains(t, string(out), "// a comment")
}

func TestNormalizeSingleMode(t *testing.T) {
	sc := &SystemConfig{Name: "robot", SystemPromptBase: "be nice"}
	normalize(sc)
	require.Equal(t, "robot", sc.DefaultMode)
	require.Len(t, sc.Modes, 1)
	mode := sc.Modes["robot"]
	require.Equal(t, "be nice", mode.SystemPromptBase)
	require.Equal(t, 1.0, mode.Hertz, "hertz defaults to 1.0")
}

func TestNormalizeFallbackModeName(t *testing.T) {
	sc := &SystemConfig{}
	normalize(sc)
	require.Contains(t, sc.Modes, "default", "with no name or default_mode, falls back to 'default'")
}

func TestNormalizePreservesExistingModes(t *testing.T) {
	sc := &SystemConfig{
		Name:  "robot",
		Modes: map[string]ModeConfig{"explicit": {Name: "explicit"}},
	}
	normalize(sc)
	require.Len(t, sc.Modes, 1)
	require.Contains(t, sc.Modes, "explicit", "existing modes are left untouched")
}

func TestResolvePath(t *testing.T) {
	require.Equal(t, "/abs/path.json5", mustResolve(t, "/abs/path.json5"), "absolute paths pass through")
	require.Equal(t, "rel/config.json", mustResolve(t, "rel/config.json"), "explicit .json suffix passes through")

	_, err := resolvePath("nonexistent-config-name")
	require.Error(t, err, "bare names that resolve to no file error out")
}

func mustResolve(t *testing.T, in string) string {
	t.Helper()
	got, err := resolvePath(in)
	require.NoError(t, err)
	return got
}

func TestLoadFullConfig(t *testing.T) {
	t.Setenv("OM1_API_KEY", "secret-key")
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json5")
	contents := `{
		// system config
		name: "tester",
		hertz: 2.0,
		api_key: "${OM1_API_KEY}",
		system_prompt_base: "be helpful",
		agent_actions: [
			{ name: "speak", connector: "tts", },
		],
	}`
	require.NoError(t, os.WriteFile(path, []byte(contents), 0o600))

	sc, err := Load(path)
	require.NoError(t, err)
	require.Equal(t, "tester", sc.Name)
	require.Equal(t, 2.0, sc.Hertz)
	require.Equal(t, "secret-key", sc.APIKey, "env var was expanded")
	require.Len(t, sc.AgentActions, 1)
	require.Equal(t, "speak", sc.AgentActions[0].Name)
	require.Contains(t, sc.Modes, "tester", "Load runs normalize")
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("/no/such/file.json5")
	require.Error(t, err)
}
