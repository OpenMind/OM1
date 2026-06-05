package memory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

const testUUID = "abc123def456abc123def456abc123ff"

func TestWriter_AppendInteraction(t *testing.T) {
	dir := t.TempDir()
	log := testLogger()

	w, err := NewWriter(dir, log)
	require.NoError(t, err)

	w.AppendInteraction("Hello robot", testUUID, "alice")
	w.AppendInteraction("What is your name?", testUUID, "alice")

	// Daily file should exist.
	dailyPath := w.dailyPath()
	content, err := os.ReadFile(dailyPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "Hello robot")
	require.Contains(t, string(content), "What is your name?")
	require.Contains(t, string(content), "[User: "+testUUID+"]")
}

func TestWriter_AppendInteraction_EmptyMessage(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	w.AppendInteraction("", testUUID, "alice")
	w.AppendInteraction("   ", testUUID, "alice")

	dailyPath := w.dailyPath()
	_, err = os.Stat(dailyPath)
	require.True(t, os.IsNotExist(err), "empty messages should not create a file")
}

func TestWriter_AppendInteraction_UnknownUser(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	w.AppendInteraction("Hello", "", "")

	content, err := os.ReadFile(w.dailyPath())
	require.NoError(t, err)
	require.Contains(t, string(content), "[User: unknown]")
}

func TestWriter_EnsureUserDir(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	w.AppendInteraction("Hi", testUUID, "bob")

	// Profile should be created with UUID.
	profilePath := filepath.Join(dir, "users", testUUID, "profile.json")
	raw, err := os.ReadFile(profilePath)
	require.NoError(t, err)

	var profile map[string]any
	require.NoError(t, json.Unmarshal(raw, &profile))
	require.Equal(t, testUUID, profile["uuid"])

	// Names should be an array containing "bob".
	names, ok := profile["names"].([]any)
	require.True(t, ok)
	require.Equal(t, []any{"bob"}, names)

	// Facts file should be created.
	factsPath := filepath.Join(dir, "users", testUUID, "facts.json")
	_, err = os.Stat(factsPath)
	require.NoError(t, err)
}

func TestWriter_UpdateUserProfile_VisitCount(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	// Multiple interactions in same session = 1 visit.
	w.AppendInteraction("Hi", testUUID, "carol")
	w.AppendInteraction("Bye", testUUID, "carol")

	profilePath := filepath.Join(dir, "users", testUUID, "profile.json")
	raw, err := os.ReadFile(profilePath)
	require.NoError(t, err)

	var profile map[string]any
	require.NoError(t, json.Unmarshal(raw, &profile))
	require.Equal(t, float64(1), profile["visit_count"], "same session = 1 visit")
	require.Equal(t, float64(2), profile["interaction_count"])
}

func TestWriter_MultipleNames(t *testing.T) {
	dir := t.TempDir()
	w, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	w.AppendInteraction("Hi", testUUID, "anon_73d0a4")
	w.AppendInteraction("Hi again", testUUID, "sean")

	profilePath := filepath.Join(dir, "users", testUUID, "profile.json")
	raw, err := os.ReadFile(profilePath)
	require.NoError(t, err)

	var profile map[string]any
	require.NoError(t, json.Unmarshal(raw, &profile))
	names, ok := profile["names"].([]any)
	require.True(t, ok)
	require.Equal(t, []any{"anon_73d0a4", "sean"}, names)
}

func TestWriter_DirectoryCreation(t *testing.T) {
	dir := t.TempDir()
	_, err := NewWriter(dir, testLogger())
	require.NoError(t, err)

	_, err = os.Stat(filepath.Join(dir, "daily"))
	require.NoError(t, err)
	_, err = os.Stat(filepath.Join(dir, "users"))
	require.NoError(t, err)
}
