package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
	r.log.Info("memory: building index...")
	if err := BuildIndex(ctx, r.index, r.dailyDir, DefaultValidDurationDays); err != nil {
		return fmt.Errorf("memory: build index: %w", err)
	}
	r.indexReady = true
	r.log.Info("memory: index initialized", zap.Int("chunks", r.index.Size()))
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

// SearchDaily searches daily logs using cosine similarity.
func (r *Reader) SearchDaily(ctx context.Context, queryText string, topK int, userID string) ([]MemoryEntry, error) {
	if strings.TrimSpace(queryText) == "" {
		return nil, nil
	}
	if topK <= 0 {
		topK = 3
	}
	return r.index.Search(ctx, queryText, topK, r.minScore, userID)
}

// FormatContext formats memory into a prompt-ready context string.
func (r *Reader) FormatContext(searchResults []MemoryEntry, maxChars int, userID string) string {
	if maxChars <= 0 {
		maxChars = defaultContextMaxChars
	}

	var parts []string
	totalChars := 0

	if userID != "" {
		userFacts := r.ReadUserFacts(userID)
		if userFacts != "" {
			section := fmt.Sprintf("[User: %s]\n%s", userID, userFacts)
			parts = append(parts, section)
			totalChars += len(section)
		}
	}

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

// ReadUserFacts reads a user's facts.json and formats as a prompt string.
func (r *Reader) ReadUserFacts(userID string) string {
	factsPath := filepath.Join(r.usersDir, userID, "facts.json")
	raw, err := os.ReadFile(factsPath)
	if err != nil {
		return ""
	}

	var data struct {
		Facts []struct {
			Fact     string `json:"fact"`
			Category string `json:"category"`
		} `json:"facts"`
	}
	if err := json.Unmarshal(raw, &data); err != nil {
		r.log.Warn("memory: failed to read facts", zap.String("user", userID), zap.Error(err))
		return ""
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
