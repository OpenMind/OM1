package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func f64(v float64) *float64  { return &v }
func strptr(v string) *string { return &v }

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
	require.Contains(t, text, "nearest first")
	require.Contains(t, text, "sean")
	require.Contains(t, text, "sean (recognized) [closest, likely speaking]")
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
	require.Contains(t, text, "sean (recognized) [closest, likely speaking]")
	require.NotContains(t, text, "anon_73d0a4 (newcomer) [closest")
}

func TestPresenceSnapshotToText_ClosestIsAnon(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "sean", UUID: "abc", Area: 800},
			{Name: "anon_73d0a4", UUID: "def", Area: 3000},
		},
	}
	text := snap.ToText()
	require.Contains(t, text, "anon_73d0a4 (newcomer) [closest, likely speaking]")
	require.NotContains(t, text, "sean (recognized) [closest")
	require.Less(t, strings.Index(text, "anon_73d0a4"), strings.Index(text, "sean"))
}

func TestPresenceSnapshotToText_ThreeFacesAnonClosest(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "doyuan", UUID: "d1", Area: 1500},
			{Name: "anon_78198", UUID: "a1", Area: 4000, LastSeenAgoSec: f64(600), LastSeenISO: strptr("2026-06-17T09:00:00Z")},
			{Name: "anon_783098", UUID: "a2", Area: 500, LastSeenAgoSec: f64(600), LastSeenISO: strptr("2026-06-17T08:00:00Z")},
		},
	}
	text := snap.ToText()

	require.Contains(t, text, "FacePresence: 3 faces")
	require.Contains(t, text, "nearest first")
	require.Contains(t, text, "anon_78198 (met before, last seen 2026-06-17) [closest, likely speaking]")
	iClosest := strings.Index(text, "anon_78198")
	iMid := strings.Index(text, "doyuan")
	iFar := strings.Index(text, "anon_783098")
	require.Less(t, iClosest, iMid)
	require.Less(t, iMid, iFar)

	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking]"))
}

func TestPresenceSnapshotToText_SingleClosestMarker(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "sean", UUID: "abc", Area: 1000}, // TrackID 0
			{Name: "kim", UUID: "def", Area: 2000},  // TrackID 0
			{Name: "lee", UUID: "ghi", Area: 500},   // TrackID 0
		},
	}
	text := snap.ToText()
	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking]"))
	require.Contains(t, text, "kim (recognized) [closest, likely speaking]")
}

func TestPresenceSnapshotToText_EqualAreaSingleMarker(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "sean", UUID: "abc", Area: 1000, TrackID: 2},
			{Name: "kim", UUID: "def", Area: 1000, TrackID: 1},
		},
	}
	text := snap.ToText()
	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking]"))
	require.Less(t, strings.Index(text, "kim"), strings.Index(text, "sean"))
	require.Contains(t, text, "kim (recognized) [closest, likely speaking]")
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
