package memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

// mockEmbedder returns a fixed-length vector for any input.
type mockEmbedder struct {
	dim int
}

func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	vec := make([]float32, m.dim)
	for i := range vec {
		vec[i] = float32(i) * 0.01
	}
	return vec, nil
}

func testLogger() *zap.Logger {
	return zap.NewNop()
}

func TestParseDailyFile(t *testing.T) {
	dir := t.TempDir()
	content := `
## 14:18:28
[User: alice]
- **User**: Hello, how are you?

## 14:19:00
[User: bob]
- **User**: What is your name?
`
	path := filepath.Join(dir, "2026-06-03.md")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	chunks, err := ParseDailyFile(path)
	require.NoError(t, err)
	require.Len(t, chunks, 2)

	require.Contains(t, chunks[0].Text, "[Date: 2026-06-03]")
	require.Contains(t, chunks[0].Text, "Hello, how are you?")
	require.Equal(t, "alice", chunks[0].Metadata["user_id"])
	require.Equal(t, "2026-06-03.md", chunks[0].Metadata["source"])

	require.Contains(t, chunks[1].Text, "What is your name?")
	require.Equal(t, "bob", chunks[1].Metadata["user_id"])
}

func TestParseDailyFile_Empty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "2026-01-01.md")
	require.NoError(t, os.WriteFile(path, []byte(""), 0o644))

	chunks, err := ParseDailyFile(path)
	require.NoError(t, err)
	require.Empty(t, chunks)
}

func TestParseDailyFile_SingleChunk(t *testing.T) {
	dir := t.TempDir()
	content := `## 10:00:00
[User: dave]
- **User**: Just one message
`
	path := filepath.Join(dir, "2026-06-04.md")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))

	chunks, err := ParseDailyFile(path)
	require.NoError(t, err)
	require.Len(t, chunks, 1)
	require.Equal(t, "dave", chunks[0].Metadata["user_id"])
}

func TestMemoryIndex_AddChunkAndSearch(t *testing.T) {
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	added, err := idx.AddChunk(context.Background(), MemoryEntry{
		Text:     "My name is Alice",
		Metadata: map[string]string{"user_id": "alice"},
	})
	require.NoError(t, err)
	require.True(t, added)
	require.Equal(t, 1, idx.Size())

	// Duplicate is no-op.
	added, err = idx.AddChunk(context.Background(), MemoryEntry{
		Text:     "My name is Alice",
		Metadata: map[string]string{"user_id": "alice"},
	})
	require.NoError(t, err)
	require.False(t, added)
	require.Equal(t, 1, idx.Size())
}

func TestMemoryIndex_LoadChunksBatch(t *testing.T) {
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	chunks := []MemoryEntry{
		{Text: "chunk one", Metadata: map[string]string{"source": "a.md"}},
		{Text: "chunk two", Metadata: map[string]string{"source": "a.md"}},
	}

	loaded, err := idx.LoadChunksBatch(context.Background(), chunks)
	require.NoError(t, err)
	require.Equal(t, 2, loaded)

	// Loading the same batch again — all should be skipped.
	loaded2, err := idx.LoadChunksBatch(context.Background(), chunks)
	require.NoError(t, err)
	require.Equal(t, 0, loaded2, "already-indexed chunks should be skipped")
	require.Equal(t, 2, idx.Size())
}

func TestMemoryIndex_SaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	_, err := idx.AddChunk(context.Background(), MemoryEntry{
		Text:     "persisted entry",
		Metadata: map[string]string{"user_id": "test"},
	})
	require.NoError(t, err)

	require.NoError(t, idx.SaveToDisk(dir))

	// Verify files exist.
	_, err = os.Stat(filepath.Join(dir, "index.graph"))
	require.NoError(t, err)
	_, err = os.Stat(filepath.Join(dir, "index.meta.json"))
	require.NoError(t, err)

	// Load into a fresh index.
	idx2 := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())
	require.NoError(t, idx2.LoadFromDisk(dir))
	require.Equal(t, 1, idx2.Size())
}

func TestMemoryIndex_PruneHashes(t *testing.T) {
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	_, _ = idx.AddChunk(context.Background(), MemoryEntry{Text: "keep me"})
	_, _ = idx.AddChunk(context.Background(), MemoryEntry{Text: "remove me"})
	require.Equal(t, 2, idx.Size())

	valid := map[string]struct{}{
		hashText("keep me"): {},
	}
	pruned := idx.PruneHashes(valid)
	require.Equal(t, 1, pruned)
	require.Equal(t, 1, idx.Size())
}

func TestBuildIndex_ExpiresOldFiles(t *testing.T) {
	dir := t.TempDir()
	dailyDir := filepath.Join(dir, "daily")
	require.NoError(t, os.MkdirAll(dailyDir, 0o755))

	// Create an old file (30 days ago).
	oldContent := "## 10:00:00\n[User: old]\n- **User**: old message\n"
	oldPath := filepath.Join(dailyDir, "2020-01-01.md")
	require.NoError(t, os.WriteFile(oldPath, []byte(oldContent), 0o644))

	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())
	require.NoError(t, BuildIndex(context.Background(), idx, dailyDir, 14))

	// Old file should be deleted.
	_, err := os.Stat(oldPath)
	require.True(t, os.IsNotExist(err))
}

func TestBuildIndex_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	// Non-existent dir is fine.
	require.NoError(t, BuildIndex(context.Background(), idx, filepath.Join(dir, "nope"), 14))
	require.Equal(t, 0, idx.Size())
}

func TestHybridSearch(t *testing.T) {
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())

	_, _ = idx.AddChunk(context.Background(), MemoryEntry{
		Text:     "David likes basketball",
		Metadata: map[string]string{"user_id": "david"},
	})
	_, _ = idx.AddChunk(context.Background(), MemoryEntry{
		Text:     "Alice works at Google",
		Metadata: map[string]string{"user_id": "alice"},
	})

	results, err := idx.HybridSearch(context.Background(), "basketball", 5, 0, "")
	require.NoError(t, err)
	require.NotEmpty(t, results)
}

func TestHybridSearch_EmptyQuery(t *testing.T) {
	idx := NewMemoryIndex(&mockEmbedder{dim: 32}, testLogger())
	_, _ = idx.AddChunk(context.Background(), MemoryEntry{Text: "something"})

	results, err := idx.Search(context.Background(), "", 5, 0, "")
	require.NoError(t, err)
	require.Nil(t, results)
}
