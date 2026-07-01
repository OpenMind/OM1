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
	summaryThreshold = 10

	defaultSummarizerModel   = "gemini-3.1-flash-lite-preview"
	defaultSummarizerBaseURL = "https://api.openmind.com/api/core/gemini"
)

var selectPrompt = strings.TrimSpace(`
Below are numbered conversation sections from a robot-human interaction log.
Identify which sections contain information worth remembering long-term
(user identity, preferences, important facts).

Output the section numbers worth remembering, one per line.
If none, respond with "NONE"
`)

var comparePrompt = strings.TrimSpace(`
Given the user's existing facts and new conversation records, integrate the new information.

For each piece of new information, output one of:
- ADD [CATEGORY] fact text
- UPDATE [CATEGORY] new fact text | replaces: old fact text
- SKIP (if already covered by existing facts)

Keep facts concise (one sentence each). If no changes needed, respond with "NONE"
`)

// Summarizer runs the LLM fact extraction + signal-based scoring pipeline.
type Summarizer struct {
	memoryRoot string
	dailyDir   string
	usersDir   string
	markerFile string
	apiKey     string
	baseURL    string
	model      string
	signals    *SignalStore
	client     *http.Client
	log        *zap.Logger

	mu      sync.Mutex
	running bool
}

// NewSummarizer creates a Summarizer.
func NewSummarizer(memoryRoot, apiKey string, signals *SignalStore, log *zap.Logger) *Summarizer {
	return &Summarizer{
		memoryRoot: memoryRoot,
		dailyDir:   filepath.Join(memoryRoot, "daily"),
		usersDir:   filepath.Join(memoryRoot, "users"),
		markerFile: filepath.Join(memoryRoot, ".last_summary"),
		apiKey:     apiKey,
		baseURL:    defaultSummarizerBaseURL,
		model:      defaultSummarizerModel,
		signals:    signals,
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

	newChunks := s.parseNewChunks(unprocessed, lastSummary)
	if len(newChunks) == 0 {
		s.writeLastSummary()
		return
	}

	promptContent := formatNumberedChunks(newChunks)
	selected, err := s.selectCandidates(ctx, promptContent)
	if err != nil {
		s.log.Error("candidate selection failed", zap.Error(err))
		return
	}

	for _, idx := range selected {
		if idx < 0 || idx >= len(newChunks) {
			continue
		}
		chunk := newChunks[idx]
		userID := parseUserFromChunk(chunk.Text)
		s.signals.MarkCandidate(chunk.Text, userID)
		s.log.Debug("marked candidate",
			zap.Int("section", idx+1),
			zap.String("user", userID),
		)
	}

	promotable := s.signals.PromotableCandidates()
	if len(promotable) > 0 {
		changedUsers := s.promoteAndSummarize(ctx, promotable)
		for _, uid := range changedUsers {
			if err := s.generateSummary(ctx, uid); err != nil {
				s.log.Warn("summary generation failed", zap.String("user", uid), zap.Error(err))
			}
		}
	}

	if expired := s.signals.ExpireCandidates(); expired > 0 {
		s.log.Info("expired candidates", zap.Int("count", expired))
	}

	s.writeLastSummary()

	s.log.Info("memory summarization complete",
		zap.Int("files", len(unprocessed)),
		zap.Int("selected", len(selected)),
		zap.Int("promotable", len(promotable)),
	)
}

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
	defer func() { _ = resp.Body.Close() }()

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

type scoredDecision struct {
	Fact     string  `json:"fact"`
	Category string  `json:"category"`
	UserID   *string `json:"user_id"`
	Decision string  `json:"decision"`
	Replaces string  `json:"replaces"`
}

// selectCandidates asks the LLM which chunks are worth remembering.
func (s *Summarizer) selectCandidates(ctx context.Context, promptContent string) ([]int, error) {
	result, err := s.llmCall(ctx, []chatMessage{
		{Role: "system", Content: selectPrompt},
		{Role: "user", Content: promptContent},
	})
	if err != nil {
		return nil, err
	}
	if strings.EqualFold(strings.TrimSpace(result), "NONE") || result == "" {
		return nil, nil
	}
	return parseSelectOutput(result), nil
}

// parseSelectOutput parses LLM output of section numbers into 0-based indices.
func parseSelectOutput(output string) []int {
	numRe := regexp.MustCompile(`^\d+$`)
	var indices []int
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if numRe.MatchString(line) {
			n := 0
			_, _ = fmt.Sscanf(line, "%d", &n)
			if n > 0 {
				indices = append(indices, n-1)
			}
		}
	}
	return indices
}

// formatNumberedChunks formats chunks as numbered sections for the LLM prompt.
func formatNumberedChunks(chunks []MemoryEntry) string {
	var parts []string
	for i, c := range chunks {
		parts = append(parts, fmt.Sprintf("[%d]\n%s", i+1, c.Text))
	}
	return strings.Join(parts, "\n\n")
}

// parseUserFromChunk extracts user ID from chunk text (e.g. "[User: alice]").
func parseUserFromChunk(text string) string {
	userRe := regexp.MustCompile(`\[User:\s*(\w+)\]`)
	m := userRe.FindStringSubmatch(text)
	if m == nil {
		return ""
	}
	return strings.ToLower(m[1])
}

// parseNewChunks parses daily logs into chunks, filtering by lastSummary.
func (s *Summarizer) parseNewChunks(files []string, lastSummary *time.Time) []MemoryEntry {
	sectionRe := regexp.MustCompile(`^## (\d{2}:\d{2}:\d{2})`)
	var result []MemoryEntry

	for _, f := range files {
		chunks, err := ParseDailyFile(f)
		if err != nil {
			continue
		}

		stem := strings.TrimSuffix(filepath.Base(f), ".md")
		fileDate, _ := time.Parse("2006-01-02", stem)

		for _, chunk := range chunks {
			if lastSummary != nil {
				lines := strings.SplitN(chunk.Text, "\n", 3)
				skip := false
				for _, line := range lines {
					m := sectionRe.FindStringSubmatch(line)
					if m != nil {
						t, err := time.Parse("15:04:05", m[1])
						if err == nil {
							sectionDT := time.Date(fileDate.Year(), fileDate.Month(), fileDate.Day(),
								t.Hour(), t.Minute(), t.Second(), 0, time.Local)
							if !sectionDT.After(*lastSummary) {
								skip = true
							}
						}
						break
					}
				}
				if skip {
					continue
				}
			}
			result = append(result, chunk)
		}
	}
	return result
}

// promoteAndSummarize takes promotable candidates, groups by user,
func (s *Summarizer) promoteAndSummarize(ctx context.Context, candidates []CandidateResult) []string {
	grouped := make(map[string][]CandidateResult)
	for _, c := range candidates {
		uid := c.UserID
		if uid == "" || uid == "unknown" {
			continue
		}
		grouped[uid] = append(grouped[uid], c)
	}

	var changedUsers []string
	for uid, chunks := range grouped {
		existingFacts := s.readUserFactsList(uid)

		var chunkTexts []string
		for _, c := range chunks {
			chunkTexts = append(chunkTexts, c.Text)
		}

		promptContent := fmt.Sprintf("Existing facts for user %q:\n%s\n\nNew conversation records:\n%s",
			uid, existingFacts, strings.Join(chunkTexts, "\n\n"))

		result, err := s.llmCall(ctx, []chatMessage{
			{Role: "system", Content: comparePrompt},
			{Role: "user", Content: promptContent},
		})
		if err != nil {
			s.log.Error("compare LLM failed", zap.String("user", uid), zap.Error(err))
			continue
		}
		if strings.EqualFold(strings.TrimSpace(result), "NONE") || result == "" {
			continue
		}

		decisions := s.parseCompareOutput(result, uid)
		if len(decisions) > 0 {
			s.applyDecisions(decisions)
			changedUsers = append(changedUsers, uid)
		}
	}
	return changedUsers
}

// readUserFactsList returns formatted existing facts for a user.
func (s *Summarizer) readUserFactsList(userID string) string {
	factsPath := filepath.Join(s.usersDir, userID, "facts.json")
	raw, err := os.ReadFile(factsPath)
	if err != nil {
		return "(no existing facts)"
	}

	var data struct {
		Facts []struct {
			Fact     string `json:"fact"`
			Category string `json:"category"`
		} `json:"facts"`
	}
	if err := json.Unmarshal(raw, &data); err != nil || len(data.Facts) == 0 {
		return "(no existing facts)"
	}

	var lines []string
	for _, f := range data.Facts {
		lines = append(lines, fmt.Sprintf("- [%s] %s", f.Category, f.Fact))
	}
	return strings.Join(lines, "\n")
}

// parseCompareOutput parses LLM ADD/UPDATE output.
func (s *Summarizer) parseCompareOutput(output, userID string) []scoredDecision {
	addRe := regexp.MustCompile(`^-?\s*ADD\s+\[(\w+)\]\s+(.+)$`)
	updateRe := regexp.MustCompile(`^-?\s*UPDATE\s+\[(\w+)\]\s+(.+?)\s*\|\s*replaces:\s*(.+)$`)

	var decisions []scoredDecision
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)

		if m := updateRe.FindStringSubmatch(line); m != nil {
			uid := userID
			decisions = append(decisions, scoredDecision{
				Category: strings.ToUpper(m[1]),
				Fact:     strings.TrimSpace(m[2]),
				Replaces: strings.TrimSpace(m[3]),
				UserID:   &uid,
				Decision: "UPDATE",
			})
		} else if m := addRe.FindStringSubmatch(line); m != nil {
			uid := userID
			decisions = append(decisions, scoredDecision{
				Category: strings.ToUpper(m[1]),
				Fact:     strings.TrimSpace(m[2]),
				UserID:   &uid,
				Decision: "PROMOTE",
			})
		}
	}
	return decisions
}

// generateSummary creates a concise natural-language summary of a user's facts.
func (s *Summarizer) generateSummary(ctx context.Context, userID string) error {
	factsPath := filepath.Join(s.usersDir, userID, "facts.json")
	raw, err := os.ReadFile(factsPath)
	if err != nil {
		return err
	}

	var data struct {
		UserID  string `json:"user_id"`
		Summary string `json:"summary"`
		Facts   []struct {
			Fact     string `json:"fact"`
			Category string `json:"category"`
			AddedAt  string `json:"added_at"`
		} `json:"facts"`
	}
	if err := json.Unmarshal(raw, &data); err != nil || len(data.Facts) == 0 {
		return nil
	}

	var factLines []string
	for _, f := range data.Facts {
		factLines = append(factLines, fmt.Sprintf("- [%s] %s", f.Category, f.Fact))
	}

	displayName := userID
	profilePath := filepath.Join(s.usersDir, userID, "profile.json")
	if pRaw, err := os.ReadFile(profilePath); err == nil {
		var profile struct {
			Names []string `json:"names"`
		}
		if json.Unmarshal(pRaw, &profile) == nil && len(profile.Names) > 0 {
			displayName = profile.Names[len(profile.Names)-1]
		}
	}

	prompt := fmt.Sprintf(
		"Summarize these facts about user %q in one concise paragraph (max 50 words):\n%s",
		displayName, strings.Join(factLines, "\n"),
	)

	summary, err := s.llmCall(ctx, []chatMessage{{Role: "user", Content: prompt}})
	if err != nil {
		return err
	}

	data.Summary = strings.TrimSpace(summary)
	out, _ := json.MarshalIndent(data, "", "  ")
	return os.WriteFile(factsPath, out, 0o644)
}

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

func (s *Summarizer) applyDecisions(decisions []scoredDecision) []string {
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
			s.log.Info("updated facts", zap.String("user", uid), zap.Int("added", added))
		}
	}

	var changedUsers []string
	for uid := range userFacts {
		changedUsers = append(changedUsers, uid)
	}
	return changedUsers
}
