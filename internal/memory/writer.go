package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go.uber.org/zap"
)

// Writer appends interactions to daily markdown files and manages
type Writer struct {
	memoryRoot string
	dailyDir   string
	usersDir   string
	log        *zap.Logger
}

// NewWriter creates a Writer, ensuring the required directories exist.
func NewWriter(memoryRoot string, log *zap.Logger) (*Writer, error) {
	dailyDir := filepath.Join(memoryRoot, "daily")
	usersDir := filepath.Join(memoryRoot, "users")

	if err := os.MkdirAll(dailyDir, 0o755); err != nil {
		return nil, fmt.Errorf("memory writer: create daily dir: %w", err)
	}
	if err := os.MkdirAll(usersDir, 0o755); err != nil {
		return nil, fmt.Errorf("memory writer: create users dir: %w", err)
	}

	return &Writer{
		memoryRoot: memoryRoot,
		dailyDir:   dailyDir,
		usersDir:   usersDir,
		log:        log,
	}, nil
}

// AppendInteraction writes a user message to today's daily log.
func (w *Writer) AppendInteraction(userMsg string, userID string) {
	if strings.TrimSpace(userMsg) == "" {
		return
	}

	if userID != "" {
		w.ensureUserDir(userID)
		w.updateUserProfile(userID)
	}

	dailyPath := w.dailyPath()
	ts := time.Now().Format("15:04:05")

	entry := fmt.Sprintf("\n## %s\n", ts)
	if userID == "" {
		userID = "unknown"
	}
	entry += fmt.Sprintf("[User: %s]\n", userID)
	entry += fmt.Sprintf("- **User**: %s\n", strings.TrimSpace(userMsg))

	f, err := os.OpenFile(dailyPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		w.log.Error("memory: failed to write interaction", zap.Error(err))
		return
	}
	defer f.Close()

	if _, err := f.WriteString(entry); err != nil {
		w.log.Error("memory: failed to write interaction", zap.Error(err))
	}
}

// AppendToIndex embeds and inserts a new user message into the given index.
func (w *Writer) AppendToIndex(ctx context.Context, idx *MemoryIndex, userMsg string, userID string) {
	if strings.TrimSpace(userMsg) == "" || idx == nil {
		return
	}

	dateStr := time.Now().Format("2006-01-02")
	ts := time.Now().Format("15:04:05")

	if userID == "" {
		userID = "unknown"
	}
	userTag := fmt.Sprintf("[User: %s]\n", userID)

	text := fmt.Sprintf("[Date: %s]\n## %s\n%s- **User**: %s",
		dateStr, ts, userTag, strings.TrimSpace(userMsg))

	meta := map[string]string{"source": dateStr + ".md", "user_id": userID}

	if _, err := idx.AddChunk(ctx, MemoryEntry{Text: text, Metadata: meta}); err != nil {
		w.log.Warn("memory: write-through index failed", zap.Error(err))
	}
}

func (w *Writer) dailyPath() string {
	today := time.Now().Format("2006-01-02")
	return filepath.Join(w.dailyDir, today+".md")
}

func (w *Writer) ensureUserDir(userID string) {
	userDir := filepath.Join(w.usersDir, userID)
	_ = os.MkdirAll(userDir, 0o755)

	profilePath := filepath.Join(userDir, "profile.json")
	if _, err := os.Stat(profilePath); os.IsNotExist(err) {
		now := time.Now().Format(time.RFC3339)
		profile := map[string]any{
			"user_id":           userID,
			"display_name":      capitalize(userID),
			"first_seen":        now,
			"last_seen":         now,
			"interaction_count": 0,
		}
		data, _ := json.MarshalIndent(profile, "", "  ")
		_ = os.WriteFile(profilePath, data, 0o644)
		w.log.Info("memory: created user profile", zap.String("user", userID))
	}

	factsPath := filepath.Join(userDir, "facts.json")
	if _, err := os.Stat(factsPath); os.IsNotExist(err) {
		facts := map[string]any{
			"user_id": userID,
			"facts":   []any{},
		}
		data, _ := json.MarshalIndent(facts, "", "  ")
		_ = os.WriteFile(factsPath, data, 0o644)
	}
}

func (w *Writer) updateUserProfile(userID string) {
	profilePath := filepath.Join(w.usersDir, userID, "profile.json")
	raw, err := os.ReadFile(profilePath)
	if err != nil {
		return
	}

	var profile map[string]any
	if err := json.Unmarshal(raw, &profile); err != nil {
		return
	}

	profile["last_seen"] = time.Now().Format(time.RFC3339)
	count, _ := profile["interaction_count"].(float64)
	profile["interaction_count"] = count + 1

	data, _ := json.MarshalIndent(profile, "", "  ")
	_ = os.WriteFile(profilePath, data, 0o644)
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
