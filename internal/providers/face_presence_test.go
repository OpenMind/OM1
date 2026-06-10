package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPresenceSnapshotToText_Empty(t *testing.T) {
	snap := PresenceSnapshot{}
	require.Equal(t, "", snap.ToText())
}

func TestPresenceSnapshotToText_OneNamed(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{{Name: "sean", UUID: "abc", Area: 1000}},
	}
	text := snap.ToText()
	require.Contains(t, text, "FacePresence: 1 face")
	require.Contains(t, text, "sean")
}

func TestPresenceSnapshotToText_NamedAndAnon(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "sean", UUID: "abc", Area: 2000},
			{Name: "anon_73d0a4", UUID: "def", Area: 1000},
		},
	}
	text := snap.ToText()
	require.Contains(t, text, "2 faces")
	require.Contains(t, text, "sean")
	require.Contains(t, text, "anon_73d0a4")
}

func TestPresenceSnapshotToText_UnknownDropped(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "unknown", Area: 1000},
			{Name: "sean", UUID: "abc", Area: 500},
		},
	}
	text := snap.ToText()
	require.Contains(t, text, "1 face")
	require.Contains(t, text, "sean")
	require.NotContains(t, text, "unknown")
}

func TestPresenceSnapshotToText_OnlyUnknown(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{{Name: "unknown", Area: 1000}},
	}
	require.Equal(t, "", snap.ToText())
}

func TestNewFacePresenceProviderDefaults(t *testing.T) {
	p := NewFacePresenceProvider(FacePresenceConfig{})
	require.Equal(t, "http://127.0.0.1:6793", p.cfg.BaseURL)
	require.Equal(t, 1.0, p.cfg.RecentSec)
	require.Equal(t, 180.0, p.cfg.AnonNewcomerThrSec)
}

func TestFetchSnapshot(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/who", r.URL.Path)
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte(`{
			"ok": true,
			"server_ts": 1700.5,
			"faces": [
				{"name": "alice", "uuid": "aaa111", "area": 2000, "tier": "confident"},
				{"name": "unknown", "area": 1500},
				{"name": "bob", "uuid": "bbb222", "area": 800, "tier": "tentative"}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL, RecentSec: 3})
	snap, err := p.FetchSnapshot(context.Background())
	require.NoError(t, err)

	require.Equal(t, 3.0, gotBody["recent_sec"], "recent_sec is sent in the request body")
	require.Len(t, snap.Faces, 3)
	require.Equal(t, "alice", snap.ClosestName, "largest UUID face is closest")
	require.Equal(t, "aaa111", snap.ClosestUUID)
}

func TestFetchSnapshotErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL})
	_, err := p.FetchSnapshot(context.Background())
	require.Error(t, err)
	require.Contains(t, err.Error(), "/who")
}
