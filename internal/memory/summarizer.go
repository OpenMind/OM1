package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	summaryThreshold = 2

	defaultSummarizerModel   = "gemini-3.1-flash-lite-preview"
	defaultSummarizerBaseURL = "https://api.openmind.com/api/core/gemini"
)

var extractPrompt = strings.TrimSpace(`
Extract candidate facts from the robot-human interaction log below.

The log may contain [User: xxx] tags indicating which user said what.
Preserve this association: if a fact comes from a tagged section,
include the user_id in your output.

For each fact, assign a category tag:
- [IDENTITY] user identity (name, age, occupation)
- [PREFERENCE] user preference (language, style, habits)
- [FACT] important facts (decisions, agreements, locations)

Output format (one per line):
- [IDENTITY] [user:alice] User's name is Alice
- [PREFERENCE] [user:bob] User prefers casual tone
- [FACT] [user:alice] User lives in Beijing
- [FACT] User asked about the weather (no user tag if unknown)

If no meaningful facts, respond with exactly "NONE"
`)

var scorePromptTmpl = strings.TrimSpace(`
You are a memory manager for a robot. Evaluate each candidate fact against the current memory and decide what to do.

Score each candidate on three dimensions (1-5):
- durability: will this matter in future conversations?
- novelty: is this new information not already in memory?
- significance: how important is this for understanding the user?

Category must be one of: IDENTITY, PREFERENCE, FACT

Decision rules:
- Total score >= 12 AND fact is not in memory → "PROMOTE"
- Fact contradicts or updates existing memory → "UPDATE" (specify which line to replace)
- Total score < 12 or already known → "SKIP"

Current memory:
%s

Candidate facts:
%s

Respond with a JSON array (no extra text):
[
  {"fact": "...", "category": "IDENTITY", "user_id": "alice", "durability": 5, "novelty": 5, "significance": 4, "decision": "PROMOTE"},
  {"fact": "...", "category": "FACT", "user_id": null, "durability": 5, "novelty": 5, "significance": 5, "decision": "UPDATE", "replaces": "old fact text"},
  {"fact": "...", "category": "PREFERENCE", "user_id": "bob", "durability": 1, "novelty": 1, "significance": 1, "decision": "SKIP"}
]

Include "user_id" if the fact is associated with a specific user, otherwise set it to null.
`)

// Summarizer runs the two-stage LLM fact extraction pipeline.
type Summarizer struct {
	memoryRoot string
	dailyDir   string
	usersDir   string
	markerFile string
	apiKey     string
	baseURL    string
	model      string
	client     *http.Client
	log        *zap.Logger

	mu      sync.Mutex
	running bool
}

// NewSummarizer creates a Summarizer.
func NewSummarizer(memoryRoot, apiKey string, log *zap.Logger) *Summarizer {
	return &Summarizer{
		memoryRoot: memoryRoot,
		dailyDir:   filepath.Join(memoryRoot, "daily"),
		usersDir:   filepath.Join(memoryRoot, "users"),
		markerFile: filepath.Join(memoryRoot, ".last_summary"),
		apiKey:     apiKey,
		baseURL:    defaultSummarizerBaseURL,
		model:      defaultSummarizerModel,
		client:     httpclient.Default(),
		log:        log,
	}
}

// CheckEligibility returns true if new conversation sections exceed the threshold.
func (s *Summarizer) CheckEligibility() bool {
	s.mu.Lock()
	running := s.running
	s.mu.Unlock()
	if running {
		return false
	}

	lastSummary := s.readLastSummary()
	unprocessed := s.findUnprocessed(lastSummary)
	if len(unprocessed) == 0 {
		return false
	}

	sectionRe := regexp.MustCompile(`^## (\d{2}:\d{2}:\d{2})`)
	count := 0
	for _, f := range unprocessed {
		stem := strings.TrimSuffix(filepath.Base(f), ".md")
		fileDate, err := time.Parse("2006-01-02", stem)
		if err != nil {
			continue
		}

		content, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		for _, line := range strings.Split(string(content), "\n") {
			m := sectionRe.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			if lastSummary != nil {
				t, err := time.Parse("15:04:05", m[1])
				if err != nil {
					count++
					continue
				}
				sectionDT := time.Date(fileDate.Year(), fileDate.Month(), fileDate.Day(),
					t.Hour(), t.Minute(), t.Second(), 0, time.Local)
				if sectionDT.After(*lastSummary) {
					count++
				}
			} else {
				count++
			}
		}
	}
	return count >= summaryThreshold
}

// Run executes the two-stage summarization pipeline.
func (s *Summarizer) Run(ctx context.Context) {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		s.running = false
		s.mu.Unlock()
	}()

	lastSummary := s.readLastSummary()
	unprocessed := s.findUnprocessed(lastSummary)
	if len(unprocessed) == 0 {
		s.writeLastSummary()
		return
	}

	logContent := s.readFiles(unprocessed, lastSummary)

	candidates, err := s.extractCandidates(ctx, logContent)
	if err != nil {
		s.log.Error("memory summarization: extract failed", zap.Error(err))
		return
	}
	if candidates == "" {
		s.writeLastSummary()
		return
	}

	decisions, err := s.scoreCandidates(ctx, candidates)
	if err != nil {
		s.log.Error("memory summarization: score failed", zap.Error(err))
		return
	}
	if len(decisions) == 0 {
		s.writeLastSummary()
		return
	}

	s.applyDecisions(decisions)
	s.writeLastSummary()

	s.log.Info("memory summarization complete",
		zap.Int("files", len(unprocessed)),
		zap.Int("candidates", len(decisions)),
	)
}

// --- LLM calls ---

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (s *Summarizer) llmCall(ctx context.Context, messages []chatMessage) (string, error) {
	body, err := json.Marshal(chatRequest{Model: s.model, Messages: messages})
	if err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.apiKey)

	resp, err := s.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("summarizer LLM returned %s: %s", resp.Status, snippet)
	}

	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", err
	}
	if len(cr.Choices) == 0 {
		return "", nil
	}
	return strings.TrimSpace(cr.Choices[0].Message.Content), nil
}

func (s *Summarizer) extractCandidates(ctx context.Context, log string) (string, error) {
	result, err := s.llmCall(ctx, []chatMessage{
		{Role: "system", Content: extractPrompt},
		{Role: "user", Content: log},
	})
	if err != nil {
		return "", err
	}
	if strings.EqualFold(result, "NONE") {
		return "", nil
	}
	return result, nil
}

type scoredDecision struct {
	Fact     string  `json:"fact"`
	Category string  `json:"category"`
	UserID   *string `json:"user_id"`
	Decision string  `json:"decision"`
	Replaces string  `json:"replaces"`
}

func (s *Summarizer) scoreCandidates(ctx context.Context, candidates string) ([]scoredDecision, error) {
	existing := s.readAllUserFacts()
	prompt := fmt.Sprintf(scorePromptTmpl, existing, candidates)

	result, err := s.llmCall(ctx, []chatMessage{
		{Role: "user", Content: prompt},
	})
	if err != nil {
		return nil, err
	}

	// Strip markdown code fences if present.
	cleaned := strings.TrimSpace(result)
	if strings.HasPrefix(cleaned, "```") {
		lines := strings.SplitN(cleaned, "\n", 2)
		if len(lines) > 1 {
			cleaned = lines[1]
		}
		if idx := strings.LastIndex(cleaned, "```"); idx >= 0 {
			cleaned = cleaned[:idx]
		}
		cleaned = strings.TrimSpace(cleaned)
	}

	var decisions []scoredDecision
	if err := json.Unmarshal([]byte(cleaned), &decisions); err != nil {
		s.log.Warn("memory summarization: failed to parse score response", zap.Error(err))
		return nil, nil
	}
	return decisions, nil
}

// --- file helpers ---

func (s *Summarizer) readLastSummary() *time.Time {
	raw, err := os.ReadFile(s.markerFile)
	if err != nil {
		return nil
	}
	t, err := time.ParseInLocation("2006-01-02 15:04", strings.TrimSpace(string(raw)), time.Local)
	if err != nil {
		return nil
	}
	return &t
}

func (s *Summarizer) writeLastSummary() {
	_ = os.WriteFile(s.markerFile, []byte(time.Now().Format("2006-01-02 15:04")), 0o644)
}

func (s *Summarizer) findUnprocessed(lastSummary *time.Time) []string {
	entries, err := os.ReadDir(s.dailyDir)
	if err != nil {
		return nil
	}

	var results []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		stem := strings.TrimSuffix(e.Name(), ".md")
		fileDate, err := time.Parse("2006-01-02", stem)
		if err != nil {
			continue
		}
		if lastSummary == nil || !fileDate.Before(lastSummary.Truncate(24*time.Hour)) {
			results = append(results, filepath.Join(s.dailyDir, e.Name()))
		}
	}
	return results
}

func (s *Summarizer) readFiles(files []string, lastSummary *time.Time) string {
	sectionRe := regexp.MustCompile(`^## (\d{2}:\d{2}:\d{2})`)
	var parts []string

	for _, f := range files {
		content, err := os.ReadFile(f)
		if err != nil {
			continue
		}

		if lastSummary == nil {
			parts = append(parts, string(content))
			continue
		}

		stem := strings.TrimSuffix(filepath.Base(f), ".md")
		fileDate, err := time.Parse("2006-01-02", stem)
		if err != nil {
			parts = append(parts, string(content))
			continue
		}

		var filtered []string
		var currentSection []string
		currentKeep := true

		for _, line := range strings.Split(string(content), "\n") {
			m := sectionRe.FindStringSubmatch(line)
			if m != nil {
				if currentKeep && len(currentSection) > 0 {
					filtered = append(filtered, strings.Join(currentSection, "\n"))
				}
				currentSection = []string{line}
				t, err := time.Parse("15:04:05", m[1])
				if err != nil {
					currentKeep = true
				} else {
					sectionDT := time.Date(fileDate.Year(), fileDate.Month(), fileDate.Day(),
						t.Hour(), t.Minute(), t.Second(), 0, time.Local)
					currentKeep = sectionDT.After(*lastSummary)
				}
			} else {
				currentSection = append(currentSection, line)
			}
		}

		if currentKeep && len(currentSection) > 0 {
			filtered = append(filtered, strings.Join(currentSection, "\n"))
		}

		if len(filtered) > 0 {
			parts = append(parts, strings.Join(filtered, "\n"))
		}
	}

	return strings.Join(parts, "\n\n")
}

func (s *Summarizer) readAllUserFacts() string {
	entries, err := os.ReadDir(s.usersDir)
	if err != nil {
		return "(no existing facts)"
	}

	var parts []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		factsPath := filepath.Join(s.usersDir, e.Name(), "facts.json")
		raw, err := os.ReadFile(factsPath)
		if err != nil {
			continue
		}

		var data struct {
			Facts []struct {
				Fact     string `json:"fact"`
				Category string `json:"category"`
			} `json:"facts"`
		}
		if err := json.Unmarshal(raw, &data); err != nil || len(data.Facts) == 0 {
			continue
		}

		lines := []string{fmt.Sprintf("[User: %s]", e.Name())}
		for _, f := range data.Facts {
			cat := f.Category
			if cat == "" {
				cat = "FACT"
			}
			lines = append(lines, fmt.Sprintf("- [%s] %s", cat, f.Fact))
		}
		parts = append(parts, strings.Join(lines, "\n"))
	}

	if len(parts) == 0 {
		return "(no existing facts)"
	}
	return strings.Join(parts, "\n\n")
}

func (s *Summarizer) applyDecisions(decisions []scoredDecision) {
	// Group by user_id.
	type factOp struct {
		fact     string
		category string
		replaces string
	}
	userFacts := make(map[string][]factOp)

	for _, d := range decisions {
		if d.Fact == "" || d.UserID == nil || *d.UserID == "" {
			continue
		}
		uid := strings.ToLower(strings.TrimSpace(*d.UserID))
		if uid == "unknown" {
			continue
		}

		switch strings.ToUpper(d.Decision) {
		case "PROMOTE":
			userFacts[uid] = append(userFacts[uid], factOp{fact: d.Fact, category: strings.ToUpper(d.Category)})
		case "UPDATE":
			userFacts[uid] = append(userFacts[uid], factOp{fact: d.Fact, category: strings.ToUpper(d.Category), replaces: d.Replaces})
		}
	}

	now := time.Now().Format(time.RFC3339)

	for uid, ops := range userFacts {
		// Ensure user dir exists.
		userDir := filepath.Join(s.usersDir, uid)
		_ = os.MkdirAll(userDir, 0o755)

		factsPath := filepath.Join(userDir, "facts.json")
		raw, err := os.ReadFile(factsPath)
		if err != nil {
			raw = []byte(fmt.Sprintf(`{"user_id": %q, "facts": []}`, uid))
		}

		var data struct {
			UserID string `json:"user_id"`
			Facts  []struct {
				Fact     string `json:"fact"`
				Category string `json:"category"`
				AddedAt  string `json:"added_at"`
			} `json:"facts"`
		}
		if err := json.Unmarshal(raw, &data); err != nil {
			data.UserID = uid
		}

		existing := make(map[string]struct{})
		for _, f := range data.Facts {
			existing[f.Fact] = struct{}{}
		}

		added := 0
		for _, op := range ops {
			if op.replaces != "" {
				// Remove old fact.
				var kept = data.Facts[:0]
				for _, f := range data.Facts {
					if f.Fact != op.replaces {
						kept = append(kept, f)
					}
				}
				data.Facts = kept
				delete(existing, op.replaces)
				s.log.Info("memory UPDATE", zap.String("user", uid), zap.String("replaced", truncate(op.replaces, 40)))
			}

			if _, dup := existing[op.fact]; !dup {
				data.Facts = append(data.Facts, struct {
					Fact     string `json:"fact"`
					Category string `json:"category"`
					AddedAt  string `json:"added_at"`
				}{
					Fact:     op.fact,
					Category: op.category,
					AddedAt:  now,
				})
				existing[op.fact] = struct{}{}
				added++
			}
		}

		if added > 0 || len(ops) > 0 {
			out, _ := json.MarshalIndent(data, "", "  ")
			_ = os.WriteFile(factsPath, out, 0o644)
			s.log.Info("memory: updated facts", zap.String("user", uid), zap.Int("added", added))
		}
	}
}
