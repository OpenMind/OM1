package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestReader_FormatContext_WithUserFacts(t *testing.T) {
	dir := t.TempDir()
	usersDir := filepath.Join(dir, "users", "alice")
	require.NoError(t, os.MkdirAll(usersDir, 0o755))

	facts := map[string]any{
		"user_id": "alice",
		"summary": "Alice is a software engineer who likes hiking.",
		"facts":   []any{},
	}
	data, _ := json.MarshalIndent(facts, "", "  ")
	require.NoError(t, os.WriteFile(filepath.Join(usersDir, "facts.json"), data, 0o644))

	reader := NewReader(dir, "", 0.5, testLogger())

	result := reader.FormatContext(nil, 500, "alice")
	require.Contains(t, result, "Alice is a software engineer")
}

func TestReader_FormatContext_WithVisitInfo(t *testing.T) {
	dir := t.TempDir()
	uuid := "deadbeef1234"
	usersDir := filepath.Join(dir, "users", uuid)
	require.NoError(t, os.MkdirAll(usersDir, 0o755))

	profile := map[string]any{
		"uuid":        uuid,
		"names":       []string{"bob"},
		"visit_count": 3,
		"last_seen":   "2026-06-03T10:00:00Z",
	}
	data, _ := json.MarshalIndent(profile, "", "  ")
	require.NoError(t, os.WriteFile(filepath.Join(usersDir, "profile.json"), data, 0o644))

	// Also need facts.json to exist (even empty).
	factsData, _ := json.MarshalIndent(map[string]any{"user_id": uuid, "facts": []any{}}, "", "  ")
	require.NoError(t, os.WriteFile(filepath.Join(usersDir, "facts.json"), factsData, 0o644))

	reader := NewReader(dir, "", 0.5, testLogger())

	result := reader.FormatContext(nil, 500, uuid)
	require.Contains(t, result, "bob")
	require.Contains(t, result, "Visited 3 times")
}

func TestReader_FormatContext_WithSearchResults(t *testing.T) {
	dir := t.TempDir()
	reader := NewReader(dir, "", 0.5, testLogger())

	results := []MemoryEntry{
		{Text: "chunk A content", Score: 0.9},
		{Text: "chunk B content", Score: 0.7},
	}

	formatted := reader.FormatContext(results, 500, "")
	require.Contains(t, formatted, "chunk A content")
	require.Contains(t, formatted, "chunk B content")
}

func TestReader_FormatContext_MaxChars(t *testing.T) {
	dir := t.TempDir()
	reader := NewReader(dir, "", 0.5, testLogger())

	results := []MemoryEntry{
		{Text: "short", Score: 0.9},
		{Text: "this is a much longer chunk that should push past the limit", Score: 0.7},
	}

	formatted := reader.FormatContext(results, 10, "")
	require.Contains(t, formatted, "short")
	require.NotContains(t, formatted, "longer chunk", "should be truncated by maxChars")
}

func TestReader_FormatContext_NoUser(t *testing.T) {
	dir := t.TempDir()
	reader := NewReader(dir, "", 0.5, testLogger())

	result := reader.FormatContext(nil, 500, "")
	require.Equal(t, "", result)
}

func TestReader_ReadUserFacts_FactsList(t *testing.T) {
	dir := t.TempDir()
	usersDir := filepath.Join(dir, "users", "eve")
	require.NoError(t, os.MkdirAll(usersDir, 0o755))

	facts := map[string]any{
		"user_id": "eve",
		"facts": []any{
			map[string]any{"fact": "Likes coffee", "category": "PREFERENCE"},
			map[string]any{"fact": "Works at OpenMind", "category": "IDENTITY"},
		},
	}
	data, _ := json.MarshalIndent(facts, "", "  ")
	require.NoError(t, os.WriteFile(filepath.Join(usersDir, "facts.json"), data, 0o644))

	reader := NewReader(dir, "", 0.5, testLogger())
	result := reader.ReadUserFacts("eve")
	require.Contains(t, result, "[PREFERENCE] Likes coffee")
	require.Contains(t, result, "[IDENTITY] Works at OpenMind")
}

func TestReader_ReadUserFacts_MissingFile(t *testing.T) {
	dir := t.TempDir()
	reader := NewReader(dir, "", 0.5, testLogger())

	result := reader.ReadUserFacts("nonexistent")
	require.Equal(t, "", result)
}
