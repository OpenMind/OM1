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

const (
	DefaultMinScore               = 0.5
	DefaultValidDurationDays      = 60
	defaultAPIURL                 = "https://api.openmind.com/api/core"
	maxUploadRound                = 1
	summarizeTriggerIntervalTicks = 10
)

type pendingRound struct {
	userMsg    string
	robotReply string
	timestamp  time.Time
}

// Manager bundles the cloud Retriever, Uploader, and local Writer.
type Manager struct {
	retriever     *Retriever
	uploader      *Uploader
	writer        *Writer
	log           *zap.Logger
	asrCount      int
	pendingUUID   string
	pendingRounds []pendingRound
}

// NewManager creates a Manager backed by cloud APIs.
func NewManager(memoryRoot, apiKey string, log *zap.Logger) *Manager {
	log = log.Named("memory")

	writer, err := NewWriter(memoryRoot, log)
	if err != nil {
		log.Warn("memory writer init failed", zap.Error(err))
	}

	m := &Manager{
		retriever: NewRetriever(defaultAPIURL, apiKey, "", log),
		uploader:  NewUploader(defaultAPIURL, apiKey, "", log),
		writer:    writer,
		log:       log,
	}

	log.Info("long-term memory enabled (cloud)", zap.String("root", memoryRoot))
	return m
}

// SearchAndFormat searches memory and returns a formatted context string.
func (m *Manager) SearchAndFormat(ctx context.Context, query string, uuid string) string {
	result, err := m.retriever.Search(ctx, query, uuid, 3, DefaultMinScore)
	if err != nil {
		m.log.Warn("memory search failed", zap.Error(err))
		return ""
	}

	// Upload recall signals
	if len(result.Chunks) > 0 {
		go m.uploader.PostSignals(context.Background(), result.Chunks, uuid)
	}

	return m.formatContext(result, uuid, 1000)
}

// RecordInteraction writes to local file and uploads to cloud.
func (m *Manager) RecordInteraction(ctx context.Context, voiceInput, robotReply, uuid, name string) {
	if m.writer != nil {
		m.writer.AppendInteraction(voiceInput, robotReply, uuid, name)
	}

	if uuid != "" {
		names := m.readLocalNames(uuid, name)
		counts := m.readLocalCounts(uuid)
		go m.uploader.PostUserProfile(context.Background(), uuid, names, counts.interactions, counts.visits)
	}

	if uuid != m.pendingUUID && len(m.pendingRounds) > 0 {
		m.uploadPendingChunk()
	}
	m.pendingUUID = uuid
	m.pendingRounds = append(m.pendingRounds, pendingRound{
		userMsg:    voiceInput,
		robotReply: robotReply,
		timestamp:  time.Now(),
	})
	if len(m.pendingRounds) >= maxUploadRound {
		m.uploadPendingChunk()
	}
}

// uploadPendingChunk merges buffered rounds into a single chunk and uploads it.
func (m *Manager) uploadPendingChunk() {
	if len(m.pendingRounds) == 0 {
		return
	}
	chunk := m.formatChunk(m.pendingUUID, m.pendingRounds)
	uuid := m.pendingUUID
	m.pendingRounds = m.pendingRounds[:0]

	go m.uploader.PostDailyLog(context.Background(), chunk, uuid)
}

func (m *Manager) formatChunk(uuid string, rounds []pendingRound) string {
	dateStr := time.Now().Format("2006-01-02")
	tag := "unknown"
	if uuid != "" {
		tag = uuid
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "[Date: %s]\n[User: %s]", dateStr, tag)
	for _, r := range rounds {
		ts := r.timestamp.Format("15:04:05")
		fmt.Fprintf(&sb, "\n[%s] User: %s", ts, strings.TrimSpace(r.userMsg))
		if reply := strings.TrimSpace(r.robotReply); reply != "" {
			fmt.Fprintf(&sb, "\n[%s] Robot: %s", ts, reply)
		}
	}
	return sb.String()
}

// Summarize triggers cloud summarization every N ASR ticks.
func (m *Manager) Summarize(ctx context.Context) {
	m.asrCount++
	if m.asrCount >= summarizeTriggerIntervalTicks {
		m.asrCount = 0
		go m.uploader.TriggerSummarize(context.Background())
	}
}

// formatContext assembles prompt context from search results.
func (m *Manager) formatContext(result *SearchResult, userID string, maxChars int) string {
	var parts []string
	totalChars := 0

	if userID != "" {
		var l1 []string

		if result.Profile != nil {
			info := formatProfileVisitInfo(result.Profile)
			if info != "" {
				l1 = append(l1, info)
			}
		}

		if result.FactsSummary != "" {
			l1 = append(l1, result.FactsSummary)
		}

		if len(l1) > 0 {
			section := strings.Join(l1, "\n")
			parts = append(parts, section)
			totalChars += len(section)
		}
	}

	for _, doc := range result.Chunks {
		if totalChars >= maxChars {
			break
		}
		if totalChars+len(doc.Text) > maxChars {
			break
		}
		parts = append(parts, doc.Text)
		totalChars += len(doc.Text)
	}

	return strings.Join(parts, "\n\n")
}

func formatProfileVisitInfo(p *ProfileData) string {
	if p.VisitCount == 0 {
		return ""
	}

	displayName := ""
	if len(p.Names) > 0 {
		displayName = p.Names[0]
		if len(p.Names) > 1 {
			displayName += " (also known as: " + strings.Join(p.Names[1:], ", ") + ")"
		}
	}

	var info string
	if displayName != "" {
		info = fmt.Sprintf("%s. Visited %d times.", displayName, p.VisitCount)
	} else {
		info = fmt.Sprintf("Visited %d times.", p.VisitCount)
	}

	if p.LastSeen != "" {
		if t, err := time.Parse(time.RFC3339, p.LastSeen); err == nil {
			days := int(time.Since(t).Hours() / 24)
			switch days {
			case 0:
				info += " Last seen: today."
			case 1:
				info += " Last seen: yesterday."
			default:
				info += fmt.Sprintf(" Last seen: %d days ago.", days)
			}
		}
	}
	return info
}

type profileCounts struct {
	interactions int
	visits       int
}

func (m *Manager) readLocalNames(uuid, name string) []string {
	if m.writer == nil {
		if name != "" {
			return []string{name}
		}
		return nil
	}
	profilePath := filepath.Join(m.writer.usersDir, uuid, "profile.json")
	raw, err := os.ReadFile(profilePath)
	if err != nil {
		if name != "" {
			return []string{name}
		}
		return nil
	}
	var data struct {
		Names []string `json:"names"`
	}
	if json.Unmarshal(raw, &data) != nil || len(data.Names) == 0 {
		if name != "" {
			return []string{name}
		}
		return nil
	}
	return data.Names
}

func (m *Manager) readLocalCounts(uuid string) profileCounts {
	if m.writer == nil {
		return profileCounts{}
	}
	profilePath := filepath.Join(m.writer.usersDir, uuid, "profile.json")
	raw, err := os.ReadFile(profilePath)
	if err != nil {
		return profileCounts{}
	}
	var data struct {
		InteractionCount float64 `json:"interaction_count"`
		VisitCount       float64 `json:"visit_count"`
	}
	if json.Unmarshal(raw, &data) != nil {
		return profileCounts{}
	}
	return profileCounts{interactions: int(data.InteractionCount), visits: int(data.VisitCount)}
}

// ResolveMemoryRoot locates the memory directory relative to cwd.
func ResolveMemoryRoot() string {
	if cwd, err := os.Getwd(); err == nil {
		candidate := filepath.Join(cwd, "memory")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
		return candidate
	}
	return "memory"
}
