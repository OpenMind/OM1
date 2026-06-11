package memory

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestSignalStore_RecordAndScore(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.Record("chunk A", 0.8, "query 1")
	store.Record("chunk A", 0.6, "query 2")

	score := store.Score("chunk A")
	require.Greater(t, score, 0.0)

	// Unknown chunk scores 0.
	require.Equal(t, 0.0, store.Score("unknown chunk"))
}

func TestSignalStore_Persistence(t *testing.T) {
	dir := t.TempDir()

	store1 := NewSignalStore(dir)
	store1.Record("persisted chunk", 0.9, "q1")
	score1 := store1.Score("persisted chunk")

	// Reload from disk.
	store2 := NewSignalStore(dir)
	score2 := store2.Score("persisted chunk")
	require.InDelta(t, score1, score2, 1e-9, "score should survive reload")
}

func TestSignalStore_InjectColdStart(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	store.InjectColdStart("new fact")
	sig := store.LookupSignal("new fact")
	require.NotNil(t, sig)
	require.Equal(t, 1, sig.RecallCount)

	// Second inject is no-op.
	store.InjectColdStart("new fact")
	sig2 := store.LookupSignal("new fact")
	require.Equal(t, 1, sig2.RecallCount)
}

func TestSignalStore_PruneStale(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	// Manually insert a stale signal.
	h := hashChunk("old chunk")
	store.mu.Lock()
	store.signals[h] = &RecallSignal{
		RecallCount:  1,
		TotalScore:   0.5,
		LastRecalled: time.Now().AddDate(0, 0, -30),
		FirstSeen:    time.Now().AddDate(0, 0, -30),
	}
	store.mu.Unlock()

	pruned := store.PruneStale(14)
	require.Equal(t, 1, pruned)
	require.Nil(t, store.LookupSignal("old chunk"))
}

func TestSignalStore_QueryHashDedup(t *testing.T) {
	dir := t.TempDir()
	store := NewSignalStore(dir)

	// Same query recorded twice — should only appear once in QueryHashes.
	store.Record("chunk", 0.5, "same query")
	store.Record("chunk", 0.5, "same query")

	sig := store.LookupSignal("chunk")
	require.NotNil(t, sig)
	require.Equal(t, 2, sig.RecallCount)
	require.Len(t, sig.QueryHashes, 1)
}

func TestComputeScore_AllDimensions(t *testing.T) {
	now := time.Now()
	sig := &RecallSignal{
		RecallCount:  5,
		TotalScore:   4.0,
		QueryHashes:  []string{"a", "b", "c"},
		RecallDays:   []string{"2026-06-01", "2026-06-02"},
		LastRecalled: now,
		FirstSeen:    now.AddDate(0, 0, -7),
	}

	score := computeScore(sig)
	require.Greater(t, score, 0.0)
	require.LessOrEqual(t, score, 1.0)
}

func TestSignalStore_FileCreated(t *testing.T) {
	dir := t.TempDir()
	NewSignalStore(dir)

	_, err := os.Stat(filepath.Join(dir, "signals"))
	require.NoError(t, err, "signals directory should be created")
}
