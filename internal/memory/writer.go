package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
)

const maxSyncPhotos = 5

// Writer appends interactions to daily markdown files and manages user profiles.
type Writer struct {
	memoryRoot         string
	dailyDir           string
	usersDir           string
	galleryDir         string
	log                *zap.Logger
	visitedThisSession map[string]struct{}
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

	var galleryDir string
	candidate := filepath.Join(filepath.Dir(memoryRoot), "gallery")
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		galleryDir = candidate
		log.Info("memory: gallery detected for photo sync", zap.String("path", galleryDir))
	}

	return &Writer{
		memoryRoot:         memoryRoot,
		dailyDir:           dailyDir,
		usersDir:           usersDir,
		galleryDir:         galleryDir,
		log:                log,
		visitedThisSession: make(map[string]struct{}),
	}, nil
}

// AppendInteraction writes a user message to today's daily log.
func (w *Writer) AppendInteraction(userMsg, uuid, name string) {
	if strings.TrimSpace(userMsg) == "" {
		return
	}

	if uuid != "" {
		w.ensureUserDir(uuid, name)
		w.updateUserProfile(uuid, name)
	}

	dailyPath := w.dailyPath()
	ts := time.Now().Format("15:04:05")

	tag := "unknown"
	if uuid != "" {
		tag = uuid
	}

	entry := fmt.Sprintf("\n## %s\n[User: %s]\n- **User**: %s\n", ts, tag, strings.TrimSpace(userMsg))

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
func (w *Writer) AppendToIndex(ctx context.Context, idx *MemoryIndex, userMsg, uuid string) {
	if strings.TrimSpace(userMsg) == "" || idx == nil {
		return
	}

	dateStr := time.Now().Format("2006-01-02")
	ts := time.Now().Format("15:04:05")

	tag := "unknown"
	if uuid != "" {
		tag = uuid
	}
	userTag := fmt.Sprintf("[User: %s]\n", tag)

	text := fmt.Sprintf("[Date: %s]\n## %s\n%s- **User**: %s",
		dateStr, ts, userTag, strings.TrimSpace(userMsg))

	meta := map[string]string{"source": dateStr + ".md", "user_id": uuid}

	if _, err := idx.AddChunk(ctx, MemoryEntry{Text: text, Metadata: meta}); err != nil {
		w.log.Warn("memory: write-through index failed", zap.Error(err))
	}
}

func (w *Writer) dailyPath() string {
	today := time.Now().Format("2006-01-02")
	return filepath.Join(w.dailyDir, today+".md")
}

func (w *Writer) ensureUserDir(uuid, name string) {
	userDir := filepath.Join(w.usersDir, uuid)
	_ = os.MkdirAll(userDir, 0o755)

	profilePath := filepath.Join(userDir, "profile.json")
	if _, err := os.Stat(profilePath); os.IsNotExist(err) {
		now := time.Now().Format(time.RFC3339)
		names := []string{}
		if name != "" {
			names = []string{name}
		}
		profile := map[string]any{
			"uuid":              uuid,
			"names":             names,
			"first_seen":        now,
			"last_seen":         now,
			"interaction_count": 0,
			"visit_count":       0,
		}
		data, _ := json.MarshalIndent(profile, "", "  ")
		_ = os.WriteFile(profilePath, data, 0o644)
		w.log.Info("memory: created user profile", zap.String("uuid", uuid), zap.Strings("names", names))
	}

	factsPath := filepath.Join(userDir, "facts.json")
	if _, err := os.Stat(factsPath); os.IsNotExist(err) {
		facts := map[string]any{
			"user_id": uuid,
			"facts":   []any{},
		}
		data, _ := json.MarshalIndent(facts, "", "  ")
		_ = os.WriteFile(factsPath, data, 0o644)
	}

	w.syncPhotos(uuid)
}

// syncPhotos copies the latest maxSyncPhotos aligned photos from the gallery
func (w *Writer) syncPhotos(uuid string) {
	if w.galleryDir == "" {
		return
	}

	srcDir := filepath.Join(w.galleryDir, uuid, "aligned")
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return
	}

	var jpgs []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".jpg") {
			jpgs = append(jpgs, e.Name())
		}
	}
	if len(jpgs) == 0 {
		return
	}
	sort.Strings(jpgs)

	if len(jpgs) > maxSyncPhotos {
		jpgs = jpgs[len(jpgs)-maxSyncPhotos:]
	}
	wanted := make(map[string]struct{}, len(jpgs))
	for _, name := range jpgs {
		wanted[name] = struct{}{}
	}

	dstDir := filepath.Join(w.usersDir, uuid, "photos")
	_ = os.MkdirAll(dstDir, 0o755)

	for _, name := range jpgs {
		dst := filepath.Join(dstDir, name)
		if _, err := os.Stat(dst); err == nil {
			continue
		}
		if err := copyFile(filepath.Join(srcDir, name), dst); err != nil {
			w.log.Warn("memory: photo sync failed", zap.String("file", name), zap.Error(err))
		}
	}

	existing, _ := os.ReadDir(dstDir)
	for _, e := range existing {
		if _, keep := wanted[e.Name()]; !keep {
			_ = os.Remove(filepath.Join(dstDir, e.Name()))
		}
	}
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

func (w *Writer) updateUserProfile(uuid, name string) {
	profilePath := filepath.Join(w.usersDir, uuid, "profile.json")
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

	if name != "" {
		names, _ := profile["names"].([]any)
		found := false
		for _, n := range names {
			if s, ok := n.(string); ok && strings.EqualFold(s, name) {
				found = true
				break
			}
		}
		if !found {
			profile["names"] = append(names, name)
		}
	}

	if _, visited := w.visitedThisSession[uuid]; !visited {
		vc, _ := profile["visit_count"].(float64)
		profile["visit_count"] = vc + 1
		w.visitedThisSession[uuid] = struct{}{}
	}

	data, _ := json.MarshalIndent(profile, "", "  ")
	_ = os.WriteFile(profilePath, data, 0o644)
}
