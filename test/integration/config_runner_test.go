//go:build integration

package integration

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/config"
)

type testCase struct {
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	WaitForCalls int      `json:"wait_for_calls"`
	TimeoutMs    int      `json:"timeout_ms"`
	Expected     expected `json:"expected"`
}

type expected struct {
	ActionCalls    []expectedCall `json:"action_calls"`
	ForbiddenCalls []expectedCall `json:"forbidden_calls"`
	MinCalls       int            `json:"min_calls"`
	NoCalls        bool           `json:"no_calls"`
}

type expectedCall struct {
	Action      string            `json:"action"`
	ArgContains map[string]string `json:"arg_contains"`
}

const testCaseDir = "testdata/test_cases"

func TestIntegrationFromConfig(t *testing.T) {
	cases, err := filepath.Glob(filepath.Join(testCaseDir, "*.json5"))
	require.NoError(t, err)
	require.NotEmpty(t, cases, "no test cases found under %s", testCaseDir)

	for _, path := range cases {
		path := path
		t.Run(strings.TrimSuffix(filepath.Base(path), ".json5"), func(t *testing.T) {
			runCaseFile(t, path)
		})
	}
}

func runCaseFile(t *testing.T, path string) {
	abs, err := filepath.Abs(path)
	require.NoError(t, err)

	sysCfg, err := config.Load(abs)
	require.NoError(t, err, "load system config")

	tc := loadTestMeta(t, abs)

	waitFor := tc.WaitForCalls
	if waitFor == 0 && !tc.Expected.NoCalls {
		waitFor = max(tc.Expected.MinCalls, len(tc.Expected.ActionCalls))
		if waitFor == 0 {
			waitFor = 1
		}
	}

	if len(tc.Expected.ForbiddenCalls) > 0 {
		waitFor = 0
	}
	timeout := 3 * time.Second
	if tc.TimeoutMs > 0 {
		timeout = time.Duration(tc.TimeoutMs) * time.Millisecond
	}

	session, s := newSession()
	calls := runRuntime(sysCfg, session, s, waitFor, timeout)

	evaluate(t, tc, calls)
}

func evaluate(t *testing.T, tc testCase, calls []recordedCall) {
	if tc.Expected.NoCalls {
		require.Empty(t, calls, "expected no actions to fire, but some did: %s", describe(calls))
		return
	}

	if tc.Expected.MinCalls > 0 {
		require.GreaterOrEqual(t, len(calls), tc.Expected.MinCalls,
			"expected at least %d actions, got %d: %s", tc.Expected.MinCalls, len(calls), describe(calls))
	}

	for _, want := range tc.Expected.ActionCalls {
		require.Truef(t, matchesAny(want, calls),
			"expected an action %q with args %v, but none matched.\nrecorded: %s",
			want.Action, want.ArgContains, describe(calls))
	}

	for _, forbidden := range tc.Expected.ForbiddenCalls {
		require.Falsef(t, matchesAny(forbidden, calls),
			"action %q with args %v should not have fired.\nrecorded: %s",
			forbidden.Action, forbidden.ArgContains, describe(calls))
	}
}

func matchesAny(want expectedCall, calls []recordedCall) bool {
	for _, got := range calls {
		if got.Label != want.Action {
			continue
		}
		if argsMatch(want.ArgContains, got.Args) {
			return true
		}
	}
	return false
}

func argsMatch(want map[string]string, got map[string]any) bool {
	for key, sub := range want {
		val, ok := got[key]
		if !ok {
			return false
		}
		if !strings.Contains(fmt.Sprintf("%v", val), sub) {
			return false
		}
	}
	return true
}

func describe(calls []recordedCall) string {
	if len(calls) == 0 {
		return "(none)"
	}
	parts := make([]string, len(calls))
	for i, c := range calls {
		parts[i] = fmt.Sprintf("%s%v", c.Label, c.Args)
	}
	return strings.Join(parts, ", ")
}

func loadTestMeta(t *testing.T, path string) testCase {
	raw, err := os.ReadFile(path)
	require.NoError(t, err)

	var tc testCase
	require.NoError(t, json.Unmarshal(stripJSON5(raw), &tc), "parse test-case metadata")
	return tc
}

func stripJSON5(src []byte) []byte {
	text := strings.ReplaceAll(string(src), "\\\n", "")
	lines := strings.Split(text, "\n")
	stripped := make([]string, 0, len(lines))
	for _, line := range lines {
		if c := commentIndex(line); c >= 0 {
			line = line[:c]
		}
		stripped = append(stripped, line)
	}
	text = strings.Join(stripped, "\n")
	text = regexp.MustCompile(`,(\s*[}\]])`).ReplaceAllString(text, "$1")
	text = regexp.MustCompile(`([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:`).ReplaceAllString(text, `$1"$2":`)
	return []byte(text)
}

func commentIndex(line string) int {
	inString := false
	for i := 0; i < len(line)-1; i++ {
		if line[i] == '"' {
			inString = !inString
		}
		if !inString && line[i] == '/' && line[i+1] == '/' {
			return i
		}
	}
	return -1
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
