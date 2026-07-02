//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/memory"
)

func embedURL() string {
	if u := os.Getenv("EMBED_URL"); u != "" {
		return u
	}
	return "http://localhost:8100"
}
func TestLocalMemory_FullPipeline(t *testing.T) {
	root := t.TempDir()
	log := zap.NewNop()
	uuid := "test_user_local"
	w, err := memory.NewWriter(root, log)
	require.NoError(t, err)

	w.AppendInteraction("What is the weather in Tokyo?", "It's 25°C and sunny.", uuid, "Alice")
	time.Sleep(100 * time.Millisecond)
	w.AppendInteraction("How about Osaka?", "Osaka is 27°C.", uuid, "Alice")
	time.Sleep(100 * time.Millisecond)
	w.AppendInteraction("Thanks, I'll pack light.", "Have a great trip!", uuid, "Alice")
	time.Sleep(100 * time.Millisecond)
	w.AppendInteraction("Hey, what's your name?", "I'm Spot!", "other_user", "Bob")
	today := time.Now().Format("2006-01-02")
	dailyDir := filepath.Join(root, "daily")
	dailyPath := filepath.Join(dailyDir, today+".md")
	content, err := os.ReadFile(dailyPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "weather in Tokyo")
	require.Contains(t, string(content), "Osaka")
	require.Contains(t, string(content), uuid)
	profilePath := filepath.Join(root, "users", uuid, "profile.json")
	profileRaw, err := os.ReadFile(profilePath)
	require.NoError(t, err)
	var profile map[string]any
	require.NoError(t, json.Unmarshal(profileRaw, &profile))
	names, _ := profile["names"].([]any)
	require.Contains(t, names, "Alice")
	chunks, err := memory.ParseDailyFile(dailyPath)
	require.NoError(t, err)
	require.Len(t, chunks, 4)
	for _, c := range chunks {
		require.False(t, c.Timestamp.IsZero(), "each chunk should have a timestamp")
	}
	idx := memory.NewIndexFromURL(embedURL(), log)
	require.NoError(t, memory.BuildIndex(context.Background(), idx, dailyDir, 60))
	require.Equal(t, 4, idx.Size())
	results, err := idx.HybridSearch(context.Background(), "weather forecast", 3, 0, uuid)
	require.NoError(t, err)
	require.NotEmpty(t, results, "should find weather-related chunks")
	topText := results[0].Text
	require.True(t,
		strings.Contains(topText, "weather") || strings.Contains(topText, "Tokyo") || strings.Contains(topText, "Osaka"),
		"top result should be semantically related to weather, got: %s", topText,
	)
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

	idx2 := memory.NewIndexFromURL(embedURL(), log)
	require.NoError(t, idx2.LoadFromDisk(indexDir))
	require.Equal(t, 4, idx2.Size())

	results2, err := idx2.HybridSearch(context.Background(), "weather forecast", 3, 0, uuid)
	require.NoError(t, err)
	enriched2 := idx2.EnrichContext(results2)
	require.NotEmpty(t, enriched2)
	require.Contains(t, enriched2[0].Text, "Tokyo")
	w.AppendToIndex(context.Background(), idx, "I also want to visit Kyoto", uuid)
	require.Equal(t, 5, idx.Size())
	results3, err := idx.HybridSearch(context.Background(), "Kyoto travel", 3, 0, uuid)
	require.NoError(t, err)
	require.NotEmpty(t, results3)
	found := false
	for _, r := range results3 {
		if strings.Contains(r.Text, "Kyoto") {
			found = true
			break
		}
	}
	require.True(t, found, "hot-updated Kyoto chunk should be searchable")
}

// mockCloudAPI mocks openmind-api memory endpoints in-memory with cosine search.
type mockCloudAPI struct {
	mu       sync.Mutex
	logs     []cloudLogEntry
	profiles []cloudProfileEntry
	signals  []json.RawMessage
}

type cloudLogEntry struct {
	FaceUUID  string    `json:"face_uuid"`
	Timestamp string    `json:"timestamp"`
	Content   string    `json:"content"`
	Embedding []float32 `json:"embedding,omitempty"`
}

type cloudProfileEntry struct {
	FaceUUID         string   `json:"face_uuid"`
	Names            []string `json:"names"`
	InteractionCount int      `json:"interaction_count"`
	VisitCount       int      `json:"visit_count"`
}

func (m *mockCloudAPI) handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /memory/daily-logs", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Entries []cloudLogEntry `json:"entries"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		m.mu.Lock()
		m.logs = append(m.logs, req.Entries...)
		m.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]int{"uploaded": len(req.Entries)})
	})

	mux.HandleFunc("POST /memory/users", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Profiles []cloudProfileEntry `json:"profiles"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		m.mu.Lock()
		m.profiles = append(m.profiles, req.Profiles...)
		m.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]int{"profiles_upserted": len(req.Profiles)})
	})

	mux.HandleFunc("POST /memory/signals", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Signals []json.RawMessage `json:"signals"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		m.mu.Lock()
		m.signals = append(m.signals, req.Signals...)
		m.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]int{"upserted": len(req.Signals)})
	})

	mux.HandleFunc("POST /memory/search", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Embedding []float32 `json:"embedding"`
			QueryText string    `json:"query_text"`
			TopK      int       `json:"top_k"`
			MinScore  float64   `json:"min_score"`
			UserID    string    `json:"user_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		m.mu.Lock()
		defer m.mu.Unlock()

		type scoredChunk struct {
			Text  string  `json:"text"`
			Score float64 `json:"score"`
		}

		var results []scoredChunk
		for _, log := range m.logs {
			if req.UserID != "" && log.FaceUUID != req.UserID {
				continue
			}
			if len(log.Embedding) == 0 || len(req.Embedding) == 0 {
				continue
			}
			score := cosineSim(req.Embedding, log.Embedding)
			if score >= req.MinScore {
				results = append(results, scoredChunk{Text: log.Content, Score: score})
			}
		}

		sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
		if len(results) > req.TopK {
			results = results[:req.TopK]
		}
		var profile *struct {
			Names      []string `json:"names"`
			VisitCount int      `json:"visit_count"`
		}
		for _, p := range m.profiles {
			if p.FaceUUID == req.UserID {
				profile = &struct {
					Names      []string `json:"names"`
					VisitCount int      `json:"visit_count"`
				}{Names: p.Names, VisitCount: p.VisitCount}
				break
			}
		}

		type chunk struct {
			Text     string            `json:"text"`
			Score    float64           `json:"score"`
			Metadata map[string]string `json:"metadata"`
		}
		chunks := make([]chunk, len(results))
		for i, r := range results {
			chunks[i] = chunk{
				Text:     r.Text,
				Score:    r.Score,
				Metadata: map[string]string{"user_id": req.UserID},
			}
		}

		resp := map[string]any{"chunks": chunks}
		if profile != nil {
			resp["profile"] = profile
		}
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("POST /memory/summarize", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	return mux
}

func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
func TestCloudMemory_FullPipeline(t *testing.T) {

	mock := &mockCloudAPI{}
	srv := httptest.NewServer(mock.handler())
	defer srv.Close()

	root := t.TempDir()
	log := zap.NewNop()
	uuid := "test_user_cloud"
	retriever := memory.NewRetriever(srv.URL, "test-key", embedURL(), log)
	uploader := memory.NewUploader(srv.URL, "test-key", embedURL(), log)

	writer, err := memory.NewWriter(root, log)
	require.NoError(t, err)
	interactions := []struct {
		userMsg    string
		robotReply string
	}{
		{"I love playing basketball on weekends", "That sounds fun!"},
		{"My favorite team is the Lakers", "Great choice!"},
		{"What's the best Italian restaurant nearby?", "Try Luigi's on 5th Ave."},
	}

	for _, inter := range interactions {

		writer.AppendInteraction(inter.userMsg, inter.robotReply, uuid, "TestUser")
		chunk := formatCloudChunk(uuid, inter.userMsg, inter.robotReply)
		uploader.PostDailyLog(context.Background(), chunk, uuid)
		time.Sleep(50 * time.Millisecond)
	}
	uploader.PostUserProfile(context.Background(), uuid, []string{"TestUser"}, len(interactions), 1)

	time.Sleep(500 * time.Millisecond) // wait for async uploads
	mock.mu.Lock()
	require.GreaterOrEqual(t, len(mock.logs), 3, "cloud should have received 3 daily logs")
	require.GreaterOrEqual(t, len(mock.profiles), 1, "cloud should have received profile")

	for _, l := range mock.logs {
		require.NotEmpty(t, l.Embedding, "each log should have an embedding vector")
		require.Equal(t, 384, len(l.Embedding), "embedding should be 384-dim (e5-small-v2)")
	}
	mock.mu.Unlock()
	result, err := retriever.Search(context.Background(), "basketball sports", uuid, 3, 0.0)
	require.NoError(t, err)
	require.NotEmpty(t, result.Chunks, "should find basketball-related chunks")

	topChunk := result.Chunks[0].Text
	require.True(t,
		strings.Contains(topChunk, "basketball") || strings.Contains(topChunk, "Lakers"),
		"top result should be basketball-related, got: %s", topChunk,
	)
	require.NotNil(t, result.Profile, "search should return user profile")
	require.Contains(t, result.Profile.Names, "TestUser")
	result2, err := retriever.Search(context.Background(), "Italian food dining", uuid, 3, 0.0)
	require.NoError(t, err)
	require.NotEmpty(t, result2.Chunks)
	topChunk2 := result2.Chunks[0].Text
	require.True(t,
		strings.Contains(topChunk2, "Italian") || strings.Contains(topChunk2, "restaurant") || strings.Contains(topChunk2, "Luigi"),
		"top result should be restaurant-related, got: %s", topChunk2,
	)
	uploader.PostSignals(context.Background(), result.Chunks, uuid, "testhash")
	time.Sleep(200 * time.Millisecond)

	mock.mu.Lock()
	require.NotEmpty(t, mock.signals, "cloud should have received recall signals")
	mock.mu.Unlock()
}
func formatCloudChunk(uuid, userMsg, robotReply string) string {
	dateStr := time.Now().Format("2006-01-02")
	ts := time.Now().Format("15:04:05")
	tag := uuid
	if tag == "" {
		tag = "unknown"
	}
	var sb strings.Builder
	sb.WriteString("[Date: " + dateStr + "]\n[User: " + tag + "]")
	sb.WriteString("\n[" + ts + "] User: " + strings.TrimSpace(userMsg))
	if reply := strings.TrimSpace(robotReply); reply != "" {
		sb.WriteString("\n[" + ts + "] Robot: " + reply)
	}
	return sb.String()
}
