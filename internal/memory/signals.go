package memory

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"
)

const (
	maxQueryHashes = 32
	maxRecallDays  = 16
	signalsFile    = "recall_signals.json"

	wFrequency     = 0.25
	wRelevance     = 0.30
	wDiversity     = 0.15
	wRecency       = 0.15
	wConsolidation = 0.15

	promotionThreshold = 0.4

	expirationRecencyThreshold = 0.1
	expirationRecallMinimum    = 2
)

// RecallSignal tracks accumulated retrieval signals for a single chunk.
type RecallSignal struct {
	RecallCount  int       `json:"recall_count"`
	TotalScore   float64   `json:"total_score"`
	QueryHashes  []string  `json:"query_hashes"`
	RecallDays   []string  `json:"recall_days"`
	LastRecalled time.Time `json:"last_recalled"`
	FirstSeen    time.Time `json:"first_seen"`
	// Candidate metadata (empty for plain chunk signals).
	Text   string `json:"text,omitempty"`
	UserID string `json:"user_id,omitempty"`
}

// CandidateResult is a promotable candidate with its score.
type CandidateResult struct {
	Text   string
	UserID string
	Score  float64
}

// SignalStore manages recall signals persisted to disk.
type SignalStore struct {
	mu       sync.Mutex
	signals  map[string]*RecallSignal
	filePath string
}

// NewSignalStore loads or creates the signal store.
func NewSignalStore(memoryRoot string) *SignalStore {
	dir := filepath.Join(memoryRoot, "signals")
	_ = os.MkdirAll(dir, 0o755)

	s := &SignalStore{
		signals:  make(map[string]*RecallSignal),
		filePath: filepath.Join(dir, signalsFile),
	}
	s.load()
	return s
}

// Record logs a recall event for a chunk.
func (s *SignalStore) Record(chunkText string, score float64, queryText string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	h := hashChunk(chunkText)
	sig, ok := s.signals[h]
	if !ok {
		sig = &RecallSignal{FirstSeen: time.Now()}
		s.signals[h] = sig
	}

	sig.RecallCount++
	sig.TotalScore += score
	sig.LastRecalled = time.Now()

	qh := hashChunk(queryText)
	if !contains(sig.QueryHashes, qh) {
		sig.QueryHashes = append(sig.QueryHashes, qh)
		if len(sig.QueryHashes) > maxQueryHashes {
			sig.QueryHashes = sig.QueryHashes[len(sig.QueryHashes)-maxQueryHashes:]
		}
	}

	today := time.Now().Format("2006-01-02")
	if !contains(sig.RecallDays, today) {
		sig.RecallDays = append(sig.RecallDays, today)
		if len(sig.RecallDays) > maxRecallDays {
			sig.RecallDays = sig.RecallDays[len(sig.RecallDays)-maxRecallDays:]
		}
	}

	s.save()
}

// Score computes the 5-dimension weighted score for a chunk.
func (s *SignalStore) Score(chunkText string) float64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	sig, ok := s.signals[hashChunk(chunkText)]
	if !ok {
		return 0
	}
	return computeScore(sig)
}

// InjectColdStart gives a new candidate a baseline signal.
func (s *SignalStore) InjectColdStart(chunkText string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	h := hashChunk(chunkText)
	if _, ok := s.signals[h]; ok {
		return
	}

	now := time.Now()
	s.signals[h] = &RecallSignal{
		RecallCount:  1,
		TotalScore:   0.5,
		QueryHashes:  []string{hashChunk("extract_" + chunkText)},
		RecallDays:   []string{now.Format("2006-01-02")},
		LastRecalled: now,
		FirstSeen:    now,
	}
	s.save()
}

// LookupSignal returns the signal for a chunk, or nil.
func (s *SignalStore) LookupSignal(chunkText string) *RecallSignal {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.signals[hashChunk(chunkText)]
}

// PruneStale removes signal entries not recalled within maxAgeDays.
func (s *SignalStore) PruneStale(maxAgeDays int) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	cutoff := time.Now().AddDate(0, 0, -maxAgeDays)
	pruned := 0
	for h, sig := range s.signals {
		if sig.LastRecalled.Before(cutoff) {
			delete(s.signals, h)
			pruned++
		}
	}
	if pruned > 0 {
		s.save()
	}
	return pruned
}

func computeScore(sig *RecallSignal) float64 {
	freq := math.Log1p(float64(sig.RecallCount)) / math.Log1p(10)
	if freq > 1 {
		freq = 1
	}

	rel := 0.0
	if sig.RecallCount > 0 {
		rel = sig.TotalScore / float64(sig.RecallCount)
	}
	if rel > 1 {
		rel = 1
	}

	uq := float64(len(sig.QueryHashes))
	ud := float64(len(sig.RecallDays))
	div := math.Max(uq, ud) / 5
	if div > 1 {
		div = 1
	}

	ageDays := time.Since(sig.LastRecalled).Hours() / 24
	rec := math.Exp(-0.693 * ageDays / 14)

	spanDays := sig.LastRecalled.Sub(sig.FirstSeen).Hours() / 24
	cons := (ud / 5) * (spanDays / 14)
	if cons > 1 {
		cons = 1
	}

	return wFrequency*freq + wRelevance*rel + wDiversity*div + wRecency*rec + wConsolidation*cons
}

func (s *SignalStore) load() {
	raw, err := os.ReadFile(s.filePath)
	if err != nil {
		return
	}
	_ = json.Unmarshal(raw, &s.signals)
}

func (s *SignalStore) save() {
	out, err := json.MarshalIndent(s.signals, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(s.filePath, out, 0o644)
}

func hashChunk(text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", h)
}

// MarkCandidate tags a chunk as a candidate in the signal store.
func (s *SignalStore) MarkCandidate(chunkText, userID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	h := hashChunk(chunkText)
	sig, ok := s.signals[h]
	if !ok {
		now := time.Now()
		sig = &RecallSignal{
			RecallCount:  1,
			TotalScore:   0.5,
			QueryHashes:  []string{hashChunk("extract_" + chunkText)},
			RecallDays:   []string{now.Format("2006-01-02")},
			LastRecalled: now,
			FirstSeen:    now,
		}
		s.signals[h] = sig
	}
	sig.Text = chunkText
	sig.UserID = userID
	s.save()
}

// PromotableCandidates returns candidates with score >= promotionThreshold.
func (s *SignalStore) PromotableCandidates() []CandidateResult {
	s.mu.Lock()
	defer s.mu.Unlock()

	var results []CandidateResult
	for _, sig := range s.signals {
		if sig.Text == "" {
			continue
		}
		score := computeScore(sig)
		if score >= promotionThreshold {
			results = append(results, CandidateResult{
				Text:   sig.Text,
				UserID: sig.UserID,
				Score:  score,
			})
		}
	}
	return results
}

// ExpireCandidates removes candidate entries where recency < 0.1 AND recall_count < 2.
func (s *SignalStore) ExpireCandidates() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	expired := 0
	for h, sig := range s.signals {
		if sig.Text == "" {
			continue
		}
		if sig.RecallCount < expirationRecallMinimum {
			ageDays := time.Since(sig.LastRecalled).Hours() / 24
			recency := math.Exp(-0.693 * ageDays / 14)
			if recency < expirationRecencyThreshold {
				delete(s.signals, h)
				expired++
			}
		}
	}
	if expired > 0 {
		s.save()
	}
	return expired
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
