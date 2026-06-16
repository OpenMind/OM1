package memory

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestMarkCandidate_New(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.MarkCandidate("chunk text here", "alice")

	sig := store.LookupSignal("chunk text here")
	require.NotNil(t, sig)
	require.Equal(t, "chunk text here", sig.Text)
	require.Equal(t, "alice", sig.UserID)
	require.Equal(t, 1, sig.RecallCount)
}

func TestMarkCandidate_ExistingSignal(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.Record("chunk text here", 0.9, "query about alice")
	store.Record("chunk text here", 0.8, "who is alice")

	store.MarkCandidate("chunk text here", "alice")

	sig := store.LookupSignal("chunk text here")
	require.NotNil(t, sig)
	require.Equal(t, 2, sig.RecallCount)
	require.Equal(t, "alice", sig.UserID)
}

func TestPromotableCandidates(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.MarkCandidate("important chunk", "bob")
	for i := 0; i < 8; i++ {
		store.Record("important chunk", 0.85, "query "+string(rune('a'+i)))
	}
	h := hashChunk("important chunk")
	store.mu.Lock()
	sig := store.signals[h]
	sig.RecallDays = []string{"2026-05-28", "2026-05-30", "2026-06-01", "2026-06-03"}
	sig.FirstSeen = time.Now().AddDate(0, 0, -7)
	store.mu.Unlock()

	store.Record("plain chunk", 0.9, "some query")

	results := store.PromotableCandidates()
	require.Len(t, results, 1)
	require.Equal(t, "important chunk", results[0].Text)
	require.Equal(t, "bob", results[0].UserID)
	require.GreaterOrEqual(t, results[0].Score, 0.75)
}

func TestPromotableCandidates_BelowThreshold(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.MarkCandidate("new chunk", "alice")

	results := store.PromotableCandidates()
	require.Empty(t, results)
}

func TestExpireCandidates(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.MarkCandidate("stale chunk", "alice")

	h := hashChunk("stale chunk")
	store.mu.Lock()
	sig := store.signals[h]
	sig.RecallCount = 1
	sig.LastRecalled = time.Now().AddDate(0, 0, -60)
	store.mu.Unlock()

	expired := store.ExpireCandidates()
	require.Equal(t, 1, expired)
	require.Nil(t, store.LookupSignal("stale chunk"))
}

func TestExpireCandidates_SkipsNonCandidates(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.Record("plain chunk", 0.5, "q")
	h := hashChunk("plain chunk")
	store.mu.Lock()
	store.signals[h].LastRecalled = time.Now().AddDate(0, 0, -60)
	store.signals[h].RecallCount = 1
	store.mu.Unlock()

	expired := store.ExpireCandidates()
	require.Equal(t, 0, expired)
	require.NotNil(t, store.LookupSignal("plain chunk"))
}

func TestParseSelectOutput(t *testing.T) {
	output := "1\n3\n5\nfoo\n\n"
	indices := parseSelectOutput(output)
	require.Equal(t, []int{0, 2, 4}, indices)
}

func TestParseSelectOutput_None(t *testing.T) {
	indices := parseSelectOutput("NONE")
	require.Empty(t, indices)
}

func TestFormatNumberedChunks(t *testing.T) {
	chunks := []MemoryEntry{
		{Text: "chunk one"},
		{Text: "chunk two"},
	}
	result := formatNumberedChunks(chunks)
	require.Contains(t, result, "[1]\nchunk one")
	require.Contains(t, result, "[2]\nchunk two")
}

func TestParseUserFromChunk(t *testing.T) {
	require.Equal(t, "alice", parseUserFromChunk("[User: alice]\n- Hello"))
	require.Equal(t, "bob", parseUserFromChunk("[User: Bob]\n- Hi"))
	require.Equal(t, "", parseUserFromChunk("no user tag here"))
}

func TestParseCompareOutput(t *testing.T) {
	s := &Summarizer{}

	output := `- ADD [IDENTITY] User's name is Alice
- UPDATE [PREFERENCE] User is now vegetarian | replaces: User likes seafood
- SKIP
- ADD [FACT] User lives in Beijing`

	decisions := s.parseCompareOutput(output, "alice")
	require.Len(t, decisions, 3)

	require.Equal(t, "PROMOTE", decisions[0].Decision)
	require.Equal(t, "User's name is Alice", decisions[0].Fact)
	require.Equal(t, "IDENTITY", decisions[0].Category)

	require.Equal(t, "UPDATE", decisions[1].Decision)
	require.Equal(t, "User is now vegetarian", decisions[1].Fact)
	require.Equal(t, "User likes seafood", decisions[1].Replaces)

	require.Equal(t, "User lives in Beijing", decisions[2].Fact)
}
