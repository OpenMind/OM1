//go:build integration

package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/knowledgebase"
	"github.com/openmind/om1/internal/memory"
)

type mockEmbedder struct{ dim int }

func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	vec := make([]float32, m.dim)
	for i := range vec {
		vec[i] = float32(i) * 0.01
	}
	return vec, nil
}

var _ knowledgebase.Embedder = (*mockEmbedder)(nil)

func TestMemory_LocalE2E(t *testing.T) {
	root := t.TempDir()
	dailyDir := filepath.Join(root, "daily")
	require.NoError(t, os.MkdirAll(dailyDir, 0o755))

	log := zap.NewNop()
	uuid := "test_user_abc123"

	today := time.Now().Format("2006-01-02")
	content := "" +
		"## 14:00:00\n[User: " + uuid + "]\n- **User**: What is the weather in Tokyo?\n- **Robot**: It's 25°C and sunny.\n\n" +
		"## 14:00:10\n[User: " + uuid + "]\n- **User**: How about Osaka?\n- **Robot**: Osaka is 27°C.\n\n" +
		"## 14:00:20\n[User: " + uuid + "]\n- **User**: Thanks, I'll pack light.\n- **Robot**: Have a great trip!\n\n" +
		"## 14:05:00\n[User: other_user]\n- **User**: Hey, what's your name?\n- **Robot**: I'm Spot!\n\n"
	require.NoError(t, os.WriteFile(filepath.Join(dailyDir, today+".md"), []byte(content), 0o644))

	chunks, err := memory.ParseDailyFile(filepath.Join(dailyDir, today+".md"))
	require.NoError(t, err)
	require.Len(t, chunks, 4)
	for _, c := range chunks {
		require.False(t, c.Timestamp.IsZero())
	}
	require.Equal(t, 10*time.Second, chunks[1].Timestamp.Sub(chunks[0].Timestamp))

	idx := memory.NewMemoryIndex(&mockEmbedder{dim: 32}, log)
	require.NoError(t, memory.BuildIndex(context.Background(), idx, dailyDir, 60))
	require.Equal(t, 4, idx.Size())

	results, err := idx.HybridSearch(context.Background(), "weather", 3, 0, uuid)
	require.NoError(t, err)
	require.NotEmpty(t, results)

	enriched := idx.EnrichContext(results)
	require.NotEmpty(t, enriched)

	aliceCount := 0
	for _, e := range enriched {
		if e.Metadata["user_id"] == uuid {
			aliceCount++
		}
	}
	require.Equal(t, 1, aliceCount, "adjacent turns should merge into 1")
	require.Contains(t, enriched[0].Text, "Tokyo")
	require.Contains(t, enriched[0].Text, "Osaka")
	require.Contains(t, enriched[0].Text, "pack light")
	require.NotContains(t, enriched[0].Text, "what's your name")

	indexDir := filepath.Join(root, "index")
	require.NoError(t, idx.SaveToDisk(indexDir))

	idx2 := memory.NewMemoryIndex(&mockEmbedder{dim: 32}, log)
	require.NoError(t, idx2.LoadFromDisk(indexDir))
	require.Equal(t, 4, idx2.Size())

	results2, err := idx2.HybridSearch(context.Background(), "weather", 3, 0, uuid)
	require.NoError(t, err)
	enriched2 := idx2.EnrichContext(results2)
	require.NotEmpty(t, enriched2)
	require.Contains(t, enriched2[0].Text, "Tokyo")
	require.Contains(t, enriched2[0].Text, "Osaka")
}

func TestMemory_WriteAndParse(t *testing.T) {
	root := t.TempDir()
	log := zap.NewNop()

	w, err := memory.NewWriter(root, log)
	require.NoError(t, err)

	uuid := "face_abc"
	w.AppendInteraction("I love sushi", "That sounds delicious!", uuid, "TestUser")
	w.AppendInteraction("Especially salmon rolls", "Great choice!", uuid, "TestUser")

	today := time.Now().Format("2006-01-02")
	dailyPath := filepath.Join(root, "daily", today+".md")
	content, err := os.ReadFile(dailyPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "I love sushi")
	require.Contains(t, string(content), "salmon rolls")
	require.Contains(t, string(content), uuid)

	profilePath := filepath.Join(root, "users", uuid, "profile.json")
	_, err = os.Stat(profilePath)
	require.NoError(t, err)

	chunks, err := memory.ParseDailyFile(dailyPath)
	require.NoError(t, err)
	require.NotEmpty(t, chunks)
	for _, c := range chunks {
		require.False(t, c.Timestamp.IsZero())
		require.Equal(t, uuid, c.Metadata["user_id"])
	}
}

func TestMemory_EnrichDeduplication(t *testing.T) {
	log := zap.NewNop()
	idx := memory.NewMemoryIndex(&mockEmbedder{dim: 32}, log)
	now := time.Now()

	for i, text := range []string{"turn A", "turn B", "turn C"} {
		_, err := idx.AddChunk(context.Background(), memory.MemoryEntry{
			Text:      text,
			Metadata:  map[string]string{"user_id": "alice"},
			Timestamp: now.Add(time.Duration(i*10) * time.Second),
		})
		require.NoError(t, err)
	}

	hits := []memory.MemoryEntry{
		{Text: "turn C", Metadata: map[string]string{"user_id": "alice"}, Timestamp: now.Add(20 * time.Second), Score: 0.9},
		{Text: "turn B", Metadata: map[string]string{"user_id": "alice"}, Timestamp: now.Add(10 * time.Second), Score: 0.8},
		{Text: "turn A", Metadata: map[string]string{"user_id": "alice"}, Timestamp: now, Score: 0.7},
	}

	enriched := idx.EnrichContext(hits)
	require.Len(t, enriched, 1)
	require.Contains(t, enriched[0].Text, "turn A")
	require.Contains(t, enriched[0].Text, "turn B")
	require.Contains(t, enriched[0].Text, "turn C")
}
