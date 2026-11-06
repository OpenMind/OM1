package config

import (
	"encoding/json"
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
	out, err := StripJSON5(src)
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
	require.Equal(t, "some/dir/agent", mustResolve(t, "some/dir/agent"), "relative paths with a separator pass through")

	_, err := resolvePath("nonexistent-config-name")
	require.Error(t, err, "bare names that resolve to no file error out")

	_, err = resolvePath("")
	require.Error(t, err, "empty config errors out")
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
		version: "v1.1.0",
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

func TestLoadUseTracerBothForms(t *testing.T) {
	dir := t.TempDir()

	boolPath := filepath.Join(dir, "bool.json5")
	require.NoError(t, os.WriteFile(boolPath, []byte(`{
		version: "v1.1.0",
		name: "tester",
		use_tracer: true,
	}`), 0o600))
	sc, err := Load(boolPath)
	require.NoError(t, err)
	require.NotNil(t, sc.UseTracer)
	require.True(t, sc.UseTracer.Enabled)
	require.Nil(t, sc.UseTracer.QualityScorer)

	objPath := filepath.Join(dir, "obj.json5")
	require.NoError(t, os.WriteFile(objPath, []byte(`{
		version: "v1.1.0",
		name: "tester",
		use_tracer: {
			enabled: true,
			quality_scorer: { enabled: true, model: "gemini-3.1-flash-lite" },
		},
	}`), 0o600))
	sc, err = Load(objPath)
	require.NoError(t, err)
	require.NotNil(t, sc.UseTracer)
	require.True(t, sc.UseTracer.Enabled)
	require.NotNil(t, sc.UseTracer.QualityScorer)
	require.True(t, sc.UseTracer.QualityScorer.Enabled)
	require.Equal(t, "gemini-3.1-flash-lite", sc.UseTracer.QualityScorer.Model)
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("/no/such/file.json5")
	require.Error(t, err)
}

func TestCommentIndexEscapedQuote(t *testing.T) {
	require.Equal(t, 9, commentIndex(`"a\"//b" // c`), "escaped quote does not end the string")
	require.Equal(t, -1, commentIndex(`"a\"//b"`), "// after an escaped quote is still inside the string")
}

func stripAndParse(t *testing.T, src string) map[string]any {
	t.Helper()
	out, err := StripJSON5([]byte(src))
	require.NoError(t, err)
	var got map[string]any
	require.NoError(t, json.Unmarshal(out, &got), "output: %s", out)
	return got
}

func TestStripJSON5PreservesStringContents(t *testing.T) {
	t.Run("trailing comma pattern inside string", func(t *testing.T) {
		got := stripAndParse(t, `{ text: "wait, }" }`)
		require.Equal(t, "wait, }", got["text"])
	})

	t.Run("comma and key-like pattern inside string", func(t *testing.T) {
		got := stripAndParse(t, `{ description: "hello, name: John" }`)
		require.Equal(t, "hello, name: John", got["description"])
	})

	t.Run("escaped quote inside string", func(t *testing.T) {
		got := stripAndParse(t, `{ quote: "say \"hi\", name: x" }`)
		require.Equal(t, `say "hi", name: x`, got["quote"])
	})

	t.Run("cleanup still applies outside strings", func(t *testing.T) {
		got := stripAndParse(t, `{ a: "x, }", b: [1, 2,], }`)
		require.Equal(t, "x, }", got["a"])
		require.Equal(t, []any{float64(1), float64(2)}, got["b"])
	})

	t.Run("quote inside a comment", func(t *testing.T) {
		got := stripAndParse(t, "{ a: 1, // it's \"x\n b: 2 }")
		require.Equal(t, float64(1), got["a"])
		require.Equal(t, float64(2), got["b"])
	})
}

func TestStripJSON5StringEdgeCases(t *testing.T) {
	t.Run("input ending with a string literal", func(t *testing.T) {
		out, err := StripJSON5([]byte(`"just a string"`))
		require.NoError(t, err)
		require.Equal(t, `"just a string"`, string(out))
	})

	t.Run("unterminated string is left untouched", func(t *testing.T) {
		out, err := StripJSON5([]byte(`{ a: "oops, }`))
		require.NoError(t, err)
		require.Equal(t, `{ "a": "oops, }`, string(out))
	})
}
