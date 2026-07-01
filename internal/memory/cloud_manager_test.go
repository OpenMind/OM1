package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func newTestCloudManager(t *testing.T) *CloudManager {
	t.Helper()
	log, _ := zap.NewDevelopment()
	return &CloudManager{log: log.Named("memory")}
}

func TestFormatChunk(t *testing.T) {
	m := newTestCloudManager(t)
	ts := time.Date(2026, 7, 1, 11, 30, 0, 0, time.UTC)
	rounds := []pendingRound{
		{userMsg: "Hello", robotReply: "Hi there!", timestamp: ts},
		{userMsg: "How are you?", robotReply: "Great!", timestamp: ts.Add(10 * time.Second)},
	}

	chunk := m.formatChunk("abc123", rounds)
	require.Contains(t, chunk, "[User: abc123]")
	require.Contains(t, chunk, "[11:30:00] User: Hello")
	require.Contains(t, chunk, "[11:30:00] Robot: Hi there!")
	require.Contains(t, chunk, "[11:30:10] User: How are you?")
	require.Contains(t, chunk, "[11:30:10] Robot: Great!")
}

func TestFormatChunk_EmptyReply(t *testing.T) {
	m := newTestCloudManager(t)
	ts := time.Date(2026, 7, 1, 9, 0, 0, 0, time.UTC)
	rounds := []pendingRound{
		{userMsg: "Hello", robotReply: "", timestamp: ts},
	}

	chunk := m.formatChunk("face1", rounds)
	require.Contains(t, chunk, "[09:00:00] User: Hello")
	require.NotContains(t, chunk, "Robot:")
}

func TestFormatChunk_EmptyUUID(t *testing.T) {
	m := newTestCloudManager(t)
	rounds := []pendingRound{
		{userMsg: "Hi", robotReply: "", timestamp: time.Now()},
	}
	chunk := m.formatChunk("", rounds)
	require.Contains(t, chunk, "[User: unknown]")
}

func TestFormatContext_WithProfile(t *testing.T) {
	m := newTestCloudManager(t)
	result := &SearchResult{
		Profile: &ProfileData{
			VisitCount: 3,
			LastSeen:   time.Now().Format(time.RFC3339),
		},
		FactsSummary: "User likes dogs.",
		Chunks: []MemoryEntry{
			{Text: "[Date: 2026-07-01]\n[User: abc]\n[11:00:00] User: Hi"},
		},
	}

	ctx := m.formatContext(result, "abc123", 2000)
	require.Contains(t, ctx, "Visited 3 times")
	require.Contains(t, ctx, "Last seen: today")
	require.Contains(t, ctx, "User likes dogs.")
	require.Contains(t, ctx, "[11:00:00] User: Hi")
}

func TestFormatContext_MaxChars(t *testing.T) {
	m := newTestCloudManager(t)
	result := &SearchResult{
		Chunks: []MemoryEntry{
			{Text: strings.Repeat("a", 100)},
			{Text: strings.Repeat("b", 100)},
		},
	}

	ctx := m.formatContext(result, "u1", 150)
	assert.Contains(t, ctx, strings.Repeat("a", 100))
	assert.NotContains(t, ctx, strings.Repeat("b", 100), "second chunk should be truncated")
}

func TestFormatContext_NoUser(t *testing.T) {
	m := newTestCloudManager(t)
	result := &SearchResult{
		Chunks: []MemoryEntry{{Text: "some data"}},
	}
	ctx := m.formatContext(result, "", 2000)
	assert.Equal(t, "some data", ctx)
}

func TestFormatProfileVisitInfo_Today(t *testing.T) {
	p := &ProfileData{
		VisitCount: 5,
		LastSeen:   time.Now().Format(time.RFC3339),
		Names:      []string{"Alice"},
	}
	info := formatProfileVisitInfo(p)
	require.Contains(t, info, "Alice. Visited 5 times.")
	require.Contains(t, info, "Last seen: today.")
}

func TestFormatProfileVisitInfo_Yesterday(t *testing.T) {
	p := &ProfileData{
		VisitCount: 2,
		LastSeen:   time.Now().Add(-30 * time.Hour).Format(time.RFC3339),
	}
	info := formatProfileVisitInfo(p)
	require.Contains(t, info, "Visited 2 times.")
	require.Contains(t, info, "Last seen: yesterday.")
}

func TestFormatProfileVisitInfo_DaysAgo(t *testing.T) {
	p := &ProfileData{
		VisitCount: 1,
		LastSeen:   time.Now().Add(-5 * 24 * time.Hour).Format(time.RFC3339),
	}
	info := formatProfileVisitInfo(p)
	require.Contains(t, info, "Last seen: 5 days ago.")
}

func TestFormatProfileVisitInfo_Zero(t *testing.T) {
	p := &ProfileData{VisitCount: 0}
	info := formatProfileVisitInfo(p)
	require.Empty(t, info)
}

func TestFormatProfileVisitInfo_MultipleNames(t *testing.T) {
	p := &ProfileData{
		VisitCount: 1,
		LastSeen:   time.Now().Format(time.RFC3339),
		Names:      []string{"Alice", "Bob"},
	}
	info := formatProfileVisitInfo(p)
	require.Contains(t, info, "Alice (also known as: Bob)")
}

func TestSummarize_TickCounter(t *testing.T) {
	// Set up a test HTTP server to capture the summarize call.
	var called atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	log, _ := zap.NewDevelopment()
	m := &CloudManager{
		uploader: &Uploader{
			apiURL: srv.URL,
			apiKey: "test",
			client: srv.Client(),
			log:    log,
		},
		log: log,
	}

	// Tick 9 times — should not trigger.
	for i := 0; i < 9; i++ {
		m.Summarize(context.Background())
	}
	require.Equal(t, 9, m.asrCount)
	time.Sleep(50 * time.Millisecond) // allow goroutine
	require.Equal(t, int32(0), called.Load())

	// 10th tick triggers.
	m.Summarize(context.Background())
	require.Equal(t, 0, m.asrCount, "should reset after trigger")
	time.Sleep(100 * time.Millisecond)
	require.Equal(t, int32(1), called.Load())
}

func TestRecordInteraction_PendingBuffer(t *testing.T) {
	var uploads atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uploads.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"uploaded":1}`))
	}))
	defer srv.Close()

	log, _ := zap.NewDevelopment()
	m := &CloudManager{
		uploader: &Uploader{
			apiURL:   srv.URL,
			apiKey:   "test",
			client:   srv.Client(),
			log:      log,
			embedder: &noopEmbedder{},
		},
		log: log,
	}

	// maxUploadRound is 1, so every call should trigger upload.
	m.RecordInteraction(context.Background(), "Hello", "Hi", "face1", "alice")
	time.Sleep(100 * time.Millisecond)
	require.Empty(t, m.pendingRounds, "buffer should be flushed")
	require.GreaterOrEqual(t, uploads.Load(), int32(1))
}

func TestRecordInteraction_UserSwitch(t *testing.T) {
	var uploadBodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := json.Marshal(map[string]int{"uploaded": 1})
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	log, _ := zap.NewDevelopment()
	m := &CloudManager{
		uploader: &Uploader{
			apiURL:   srv.URL,
			apiKey:   "test",
			client:   srv.Client(),
			log:      log,
			embedder: &noopEmbedder{},
		},
		log: log,
	}
	_ = uploadBodies

	m.RecordInteraction(context.Background(), "Hi from user1", "Hello", "user1", "")
	require.Equal(t, "user1", m.pendingUUID)
}

func TestRecordInteraction_EmptyUUID(t *testing.T) {
	log, _ := zap.NewDevelopment()
	m := &CloudManager{log: log}

	m.RecordInteraction(context.Background(), "Hello", "Hi", "", "")
	require.Empty(t, m.pendingRounds, "empty UUID should skip cloud upload")
}

func TestSearchAndFormat_EmptyUUID(t *testing.T) {
	m := newTestCloudManager(t)
	result := m.SearchAndFormat(context.Background(), "hello", "")
	require.Empty(t, result)
}

// noopEmbedder returns a zero vector for testing.
type noopEmbedder struct{}

func (e *noopEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, 8), nil
}
