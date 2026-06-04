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

func TestJoinNames(t *testing.T) {
	require.Equal(t, "", joinNames(nil))
	require.Equal(t, "alice", joinNames([]string{"alice"}))
	require.Equal(t, "alice and bob", joinNames([]string{"alice", "bob"}))
	require.Equal(t, "alice, bob and carol", joinNames([]string{"alice", "bob", "carol"}))
}

func TestPresenceSnapshotToText(t *testing.T) {
	cases := []struct {
		name string
		snap PresenceSnapshot
		want string
	}{
		{"nobody", PresenceSnapshot{}, "No one in view."},
		{
			"one known",
			PresenceSnapshot{Names: []string{"wendy"}, ClosestName: "wendy"},
			"In Camera View: 1 known (wendy). Closest: wendy.",
		},
		{
			"two known plus unknowns",
			PresenceSnapshot{Names: []string{"wendy", "alice"}, UnknownFaces: 2, ClosestName: "wendy"},
			"In Camera View: 2 known (wendy and alice) and 2 unknown faces. Closest: wendy.",
		},
		{
			"single unknown",
			PresenceSnapshot{UnknownFaces: 1, ClosestName: "unknown"},
			"In Camera View: 1 unknown face. Closest: unknown.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, tc.snap.ToText())
		})
	}
}

func TestPresenceSnapshotToTextDedupsAndDropsUnknown(t *testing.T) {
	snap := PresenceSnapshot{Names: []string{"wendy", "wendy", "unknown", " "}, ClosestName: "wendy"}
	require.Equal(t, "In Camera View: 1 known (wendy). Closest: wendy.", snap.ToText(),
		"duplicate, blank, and 'unknown' names are filtered from the known list")
}

func TestNewFacePresenceProviderDefaults(t *testing.T) {
	p := NewFacePresenceProvider(FacePresenceConfig{})
	require.Equal(t, "http://127.0.0.1:6793", p.baseURL)
	require.Equal(t, 1.0, p.recentSec)
	require.Equal(t, 2*time.Second, p.timeout)
	require.Equal(t, 500.0, p.minFaceArea)
}

func TestNewFacePresenceProviderTrimsSlash(t *testing.T) {
	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: "http://host:1/"})
	require.Equal(t, "http://host:1", p.baseURL)
}

func TestFetchSnapshot(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/who", r.URL.Path)
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte(`{
			"server_ts": 1700.5,
			"faces": [
				{"name": "alice", "area": 2000},
				{"name": "unknown", "area": 1500},
				{"name": "tiny", "area": 10},
				{"name": "bob", "area": 800}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL, MinFaceArea: 500, RecentSec: 3})
	snap, err := p.FetchSnapshot(context.Background())
	require.NoError(t, err)

	require.Equal(t, 3.0, gotBody["recent_sec"], "recent_sec is sent in the request body")
	require.Equal(t, 1700.5, snap.Timestamp)
	require.Equal(t, []string{"alice", "bob"}, snap.Names, "ordered by area desc, tiny face dropped, unknown excluded from names")
	require.Equal(t, 1, snap.UnknownFaces, "the large unknown face is counted")
	require.Equal(t, "alice", snap.ClosestName, "largest face is closest")
}

func TestFetchSnapshotErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	p := NewFacePresenceProvider(FacePresenceConfig{BaseURL: srv.URL})
	_, err := p.FetchSnapshot(context.Background())
	require.Error(t, err)
	require.Contains(t, err.Error(), "500")
}
