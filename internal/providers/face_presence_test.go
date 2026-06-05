package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestNewFacePresenceProviderDefaults(t *testing.T) {
	p := NewFacePresenceProvider(FacePresenceConfig{})
	require.Equal(t, "http://127.0.0.1:6793", p.cfg.BaseURL)
	require.Equal(t, 1.0, p.cfg.RecentSec)
	require.Equal(t, 2*time.Second, p.cfg.Timeout)
	require.Equal(t, 180.0, p.cfg.AnonNewcomerThrSec)
	require.Equal(t, 86_400.0, p.cfg.NamedRecentThrSec)
	require.Equal(t, 604_800.0, p.cfg.NamedLongAbsenceThrSec)
}



func TestPresenceSnapshotToTextEmpty(t *testing.T) {
	snap := PresenceSnapshot{}
	require.Equal(t, "", snap.ToText(), "no faces → empty string")
}

func TestPresenceSnapshotToTextNamedOnly(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "wendy", Area: 2000},
		},
	}
	got := snap.ToText()
	require.Contains(t, got, "FacePresence: 1 face")
	require.Contains(t, got, "wendy (recognized)")
}

func TestPresenceSnapshotToTextDropsUnknown(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "wendy", Area: 2000},
			{Name: "unknown", Area: 1500},
			{Name: "", Area: 800},
		},
	}
	got := snap.ToText()
	// Only wendy should count — unknown and blank names are dropped.
	require.Contains(t, got, "FacePresence: 1 face")
	require.Contains(t, got, "wendy (recognized)")
	require.NotContains(t, got, "unknown")
}

func TestPresenceSnapshotToTextAnon(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "anon_73d0a4", Area: 2000},
		},
	}
	got := snap.ToText()
	require.Contains(t, got, "FacePresence: 1 face")
	require.Contains(t, got, "anon_73d0a4 (newcomer)")
}

func TestPresenceSnapshotToTextMixed(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "wendy", Area: 3000},
			{Name: "anon_abc123", Area: 1500},
		},
	}
	got := snap.ToText()
	require.Contains(t, got, "FacePresence: 2 faces")
	require.Contains(t, got, "wendy (recognized)")
	require.Contains(t, got, "anon_abc123 (newcomer)")
}

func TestPresenceSnapshotToTextOnlyUnknowns(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "unknown", Area: 2000},
		},
	}
	require.Equal(t, "", snap.ToText(), "only unknown faces → empty string")
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
				{"name": "alice", "uuid": "aaa", "area": 2000, "tier": "confident", "sim": 0.85},
				{"name": "unknown", "area": 1500},
				{"name": "anon_xyz", "uuid": "bbb", "area": 800, "tier": "tentative", "sim": 0.6},
				{"name": "bob", "uuid": "ccc", "area": 1800, "tier": "confident", "sim": 0.9}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL, RecentSec: 3})
	snap, err := p.FetchSnapshot(context.Background())
	require.NoError(t, err)

	require.Equal(t, 3.0, gotBody["recent_sec"], "recent_sec is sent in the request body")
	require.Equal(t, 1700.5, snap.ServerTS)
	require.True(t, snap.OK)
	require.Len(t, snap.Faces, 4)

	// Closest* derived fields should reflect the largest-area face with a UUID.
	require.Equal(t, "alice", snap.ClosestName, "largest UUID face is closest")
	require.Equal(t, "aaa", snap.ClosestUUID)
	require.Equal(t, "confident", snap.ClosestTier)
	require.Equal(t, 0.85, snap.ClosestSim)
}

func TestFetchSnapshotClosestAnonLeavesNameEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"ok": true,
			"faces": [
				{"name": "anon_abc", "uuid": "xyz", "area": 5000, "tier": "tentative", "sim": 0.5}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL})
	snap, err := p.FetchSnapshot(context.Background())
	require.NoError(t, err)
	require.Equal(t, "", snap.ClosestName, "anon closest → empty ClosestName")
	require.Equal(t, "xyz", snap.ClosestUUID)
}

func TestFetchSnapshotErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL})
	_, err := p.FetchSnapshot(context.Background())
	require.Error(t, err, "500 response should cause a decode error")
}

func TestFormatNamedEntryTiers(t *testing.T) {
	// < 1 day (or nil) → "(recognized)"
	f := FaceEntry{Name: "sean"}
	require.Equal(t, "sean (recognized)", formatNamedEntry(f))

	// < 1 day explicitly
	gap := 3600.0 // 1 hour
	f = FaceEntry{Name: "sean", LastSeenAgoSec: &gap}
	require.Equal(t, "sean (recognized)", formatNamedEntry(f))

	// 1-6 days → "(recognized, N days ago)"
	gap = 172_800.0 // 2 days
	f = FaceEntry{Name: "sean", LastSeenAgoSec: &gap}
	require.Equal(t, "sean (recognized, 2 days ago)", formatNamedEntry(f))

	// >= 7 days → "(recognized, last seen DATE)"
	gap = 700_000.0
	iso := "2026-01-15T10:00:00"
	f = FaceEntry{Name: "sean", LastSeenAgoSec: &gap, LastSeenISO: &iso}
	require.Equal(t, "sean (recognized, last seen 2026-01-15)", formatNamedEntry(f))
}

func TestFormatAnonEntryTiers(t *testing.T) {
	// nil → newcomer
	f := FaceEntry{Name: "anon_abc"}
	require.Equal(t, "anon_abc (newcomer)", formatAnonEntry(f))

	// < 3 min → newcomer
	gap := 60.0
	f = FaceEntry{Name: "anon_abc", LastSeenAgoSec: &gap}
	require.Equal(t, "anon_abc (newcomer)", formatAnonEntry(f))

	// >= 3 min → met before
	gap = 300.0
	f = FaceEntry{Name: "anon_abc", LastSeenAgoSec: &gap}
	require.Equal(t, "anon_abc (met before)", formatAnonEntry(f))

	// >= 3 min with ISO → met before, last seen DATE
	iso := "2026-06-04T09:00:00"
	f = FaceEntry{Name: "anon_abc", LastSeenAgoSec: &gap, LastSeenISO: &iso}
	require.Equal(t, "anon_abc (met before, last seen 2026-06-04)", formatAnonEntry(f))
}

func TestLastSeenDate(t *testing.T) {
	require.Equal(t, "", lastSeenDate(nil))
	empty := ""
	require.Equal(t, "", lastSeenDate(&empty))

	iso := "2026-03-05T14:30:00"
	require.Equal(t, "2026-03-05", lastSeenDate(&iso))

	// With extra timezone suffix
	iso = "2026-03-05T14:30:00+00:00"
	require.Equal(t, "2026-03-05", lastSeenDate(&iso))
}
