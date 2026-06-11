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

	"github.com/openmind/om1/internal/knowledgebase"
)

const defaultContextMaxChars = 1000

// Reader searches and reads long-term memory files.
type Reader struct {
	MemoryRoot string
	dailyDir   string
	usersDir   string
	index      *MemoryIndex
	indexReady bool
	minScore   float64
	log        *zap.Logger
}

// NewReader creates a Reader. Call EnsureIndex before searching.
func NewReader(memoryRoot string, embedderBaseURL string, minScore float64, log *zap.Logger) *Reader {
	if minScore <= 0 {
		minScore = DefaultMinScore
	}
	embedder := knowledgebase.NewHTTPEmbedder(embedderBaseURL)
	return &Reader{
		MemoryRoot: memoryRoot,
		dailyDir:   filepath.Join(memoryRoot, "daily"),
		usersDir:   filepath.Join(memoryRoot, "users"),
		index:      NewMemoryIndex(embedder, log),
		minScore:   minScore,
		log:        log,
	}
}

// EnsureIndex builds the index from daily files on first call.
func (r *Reader) EnsureIndex(ctx context.Context) error {
	if r.indexReady {
		return nil
	}
	r.log.Info("building index...")
	if err := BuildIndex(ctx, r.index, r.dailyDir, DefaultValidDurationDays); err != nil {
		return fmt.Errorf("memory: build index: %w", err)
	}
	r.indexReady = true
	r.log.Info("index initialized", zap.Int("chunks", r.index.Size()))
	return nil
}

// Index returns the underlying MemoryIndex (used by Writer for hot updates).
func (r *Reader) Index() *MemoryIndex {
	return r.index
}

// IndexReady reports whether the index has been initialized.
func (r *Reader) IndexReady() bool {
	return r.indexReady
}

// SearchDaily searches daily logs using hybrid retrieval (embedding + BM25).
func (r *Reader) SearchDaily(ctx context.Context, queryText string, topK int, userID string) ([]MemoryEntry, error) {
	if strings.TrimSpace(queryText) == "" {
		return nil, nil
	}
	if topK <= 0 {
		topK = 3
	}
	return r.index.HybridSearch(ctx, queryText, topK, r.minScore, userID)
}

// FormatContext assembles memory context for prompt injection.
// L1: User profile + facts summary — when user_id is available
// L2: Hybrid search results — fills remaining budget
func (r *Reader) FormatContext(searchResults []MemoryEntry, maxChars int, userID string) string {
	if maxChars <= 0 {
		maxChars = defaultContextMaxChars
	}

	var parts []string
	totalChars := 0

	// L1: User-specific context.
	if userID != "" {
		var l1Parts []string

		// Profile visit info.
		visitInfo := r.readProfileVisitInfo(userID)
		if visitInfo != "" {
			l1Parts = append(l1Parts, visitInfo)
		}

		// Facts summary.
		userFacts := r.ReadUserFacts(userID)
		if userFacts != "" {
			l1Parts = append(l1Parts, userFacts)
		}

		if len(l1Parts) > 0 {
			section := strings.Join(l1Parts, "\n")
			parts = append(parts, section)
			totalChars += len(section)
		}
	}

	// L2: Relevant history from hybrid search.
	for _, doc := range searchResults {
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

// readProfileVisitInfo reads profile.json and formats visit info.
func (r *Reader) readProfileVisitInfo(uuid string) string {
	profilePath := filepath.Join(r.usersDir, uuid, "profile.json")
	raw, err := os.ReadFile(profilePath)
	if err != nil {
		return ""
	}

	var profile struct {
		Names      []string `json:"names"`
		VisitCount int      `json:"visit_count"`
		LastSeen   string   `json:"last_seen"`
	}
	if err := json.Unmarshal(raw, &profile); err != nil {
		return ""
	}

	if profile.VisitCount == 0 {
		return ""
	}

	displayName := ""
	if len(profile.Names) > 0 {
		displayName = profile.Names[0]
		if len(profile.Names) > 1 {
			displayName += " (also known as: " + strings.Join(profile.Names[1:], ", ") + ")"
		}
	}

	var info string
	if displayName != "" {
		info = fmt.Sprintf("%s. Visited %d times.", displayName, profile.VisitCount)
	} else {
		info = fmt.Sprintf("Visited %d times.", profile.VisitCount)
	}

	if profile.LastSeen != "" {
		if t, err := time.Parse(time.RFC3339, profile.LastSeen); err == nil {
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

// ReadUserFacts reads a user's facts.json and returns the summary field.
func (r *Reader) ReadUserFacts(userID string) string {
	factsPath := filepath.Join(r.usersDir, userID, "facts.json")
	raw, err := os.ReadFile(factsPath)
	if err != nil {
		return ""
	}

	var data struct {
		Summary string `json:"summary"`
		Facts   []struct {
			Fact     string `json:"fact"`
			Category string `json:"category"`
		} `json:"facts"`
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		r.log.Warn("failed to read facts", zap.String("user", userID), zap.Error(err))
		return ""
	}

	if data.Summary != "" {
		return data.Summary
	}

	if len(data.Facts) == 0 {
		return ""
	}
	var lines []string
	for _, f := range data.Facts {
		if f.Fact == "" {
			continue
		}
		cat := f.Category
		if cat == "" {
			cat = "FACT"
		}
		lines = append(lines, fmt.Sprintf("- [%s] %s", cat, f.Fact))
	}
	return strings.Join(lines, "\n")
}
