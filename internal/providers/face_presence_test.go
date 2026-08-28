package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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
	require.Contains(t, text, "sean (recognized) [closest, likely speaking")
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
	require.Contains(t, text, "sean (recognized) [closest, likely speaking")
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
	require.Contains(t, text, "anon_73d0a4 (newcomer) [closest, likely speaking")
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
	require.Contains(t, text, "anon_78198 (met before, last seen 2026-06-17) [closest, likely speaking")
	iClosest := strings.Index(text, "anon_78198")
	iMid := strings.Index(text, "doyuan")
	iFar := strings.Index(text, "anon_783098")
	require.Less(t, iClosest, iMid)
	require.Less(t, iMid, iFar)

	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking"))
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
	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking"))
	require.Contains(t, text, "kim (recognized) [closest, likely speaking")
}

func TestPresenceSnapshotToText_EqualAreaSingleMarker(t *testing.T) {
	snap := PresenceSnapshot{
		Faces: []FaceEntry{
			{Name: "sean", UUID: "abc", Area: 1000, TrackID: 2},
			{Name: "kim", UUID: "def", Area: 1000, TrackID: 1},
		},
	}
	text := snap.ToText()
	require.Equal(t, 1, strings.Count(text, "[closest, likely speaking"))
	require.Less(t, strings.Index(text, "kim"), strings.Index(text, "sean"))
	require.Contains(t, text, "kim (recognized) [closest, likely speaking")
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
	// Assert on the face list only. "unknown" now legitimately appears in
	// the trailing Speaker line, which is a statement about who was heard,
	// not about which faces are listed.
	faceList := strings.SplitN(text, "\nSpeaker:", 2)[0]
	require.NotContains(t, faceList, "unknown")
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

// --- speaker attribution -------------------------------------------------
//
// The case these exist for: the person talking is NOT the biggest face.
// Proximity picks the wrong one, and everything the model is told in that
// turn -- a name above all -- lands on the wrong identity.

func twoFaces() *PresenceSnapshot {
	return &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "jan", UUID: "j1", Area: 9000, TrackID: 61},   // nearest
		{Name: "wendy", UUID: "w1", Area: 3000, TrackID: 54}, // actually talking
	}}
}

func TestToText_SpeakerBeatsProximity(t *testing.T) {
	snap := twoFaces()
	spk := &SpeakerResult{TrackID: 54, Name: "wendy", UUID: "w1", Score: 0.87, ResolvedAt: timeNow()}

	text := snap.toTextWithSpeaker(spk, true)

	require.Contains(t, text, "wendy (recognized) [SPEAKING NOW]")
	require.Contains(t, text, "jan (recognized) [not speaking]")
	require.Contains(t, text, "Speaker: wendy (track 54, confidence 0.87)")
	require.Contains(t, text, "attribute what was just said to THIS person")
	// The proximity guess must be gone once a measurement exists, or the
	// model gets two contradictory hints and picks whichever it likes.
	require.NotContains(t, text, "likely speaking")
}

func TestToText_NoSpeakerFallsBackToGuess(t *testing.T) {
	text := twoFaces().toTextWithSpeaker(nil, true)

	require.Contains(t, text, "jan (recognized) [closest, likely speaking — GUESS, no speech detection]")
	require.Contains(t, text, "Speaker: unknown (no utterance resolved yet)")
	require.NotContains(t, text, "SPEAKING NOW")
}

func TestToText_DetectionUnavailableSaysSo(t *testing.T) {
	text := twoFaces().toTextWithSpeaker(nil, false)

	require.Contains(t, text, "no active-speaker detection running")
	require.Contains(t, text, "do NOT assume the nearest face is the one talking")
}

func TestToText_NobodyScoredAsSpeaking(t *testing.T) {
	spk := &SpeakerResult{TrackID: -1, ResolvedAt: timeNow()}
	text := twoFaces().toTextWithSpeaker(spk, true)

	require.Contains(t, text, "nobody scored as speaking")
	require.NotContains(t, text, "SPEAKING NOW")
}

func TestToText_SpeakerNoLongerVisible(t *testing.T) {
	// Resolved to a track that has since been dropped: say so rather than
	// silently reassigning the utterance to whoever is on screen now.
	spk := &SpeakerResult{TrackID: 99, Name: "wendy", UUID: "w1", Score: 0.8, ResolvedAt: timeNow()}
	text := twoFaces().toTextWithSpeaker(spk, true)

	require.Contains(t, text, "no longer visible")
	require.NotContains(t, text, "SPEAKING NOW")
}

func TestToText_EnrolledButUnnamedSpeaker(t *testing.T) {
	spk := &SpeakerResult{TrackID: 54, Name: "anon_73d0a4", UUID: "w1", Score: 0.7, ResolvedAt: timeNow()}
	text := twoFaces().toTextWithSpeaker(spk, true)

	// "unnamed" and "unrecognised" are different situations: the first can
	// be given a name right now, the second cannot.
	require.Contains(t, text, "an enrolled but unnamed person")
}

// timeNow is a stable stand-in for time.Now in these fixtures.
func timeNow() time.Time { return time.Now() }

// --- memory attribution --------------------------------------------------
//
// Who the turn gets FILED under, which is a different question from who is
// tagged on screen and has a different failure mode: a wrong tag is visible,
// a wrong filing silently accumulates one person's history under another's.

func TestAttributedUser_FollowsSpeakerNotProximity(t *testing.T) {
	Speaker().Reset()
	defer Speaker().Reset()

	snap := twoFaces()
	snap.ClosestUUID, snap.ClosestName = "j1", "jan" // what proximity would pick

	Speaker().mu.Lock()
	Speaker().latest = &SpeakerResult{TrackID: 54, Name: "wendy", UUID: "w1", ResolvedAt: time.Now()}
	Speaker().mu.Unlock()

	uuid, name, measured := snap.AttributedUser()
	require.Equal(t, "w1", uuid, "the turn belongs to whoever spoke")
	require.Equal(t, "wendy", name)
	require.True(t, measured)
}

func TestAttributedUser_UnnamedPersonStillGetsRecorded(t *testing.T) {
	Speaker().Reset()
	defer Speaker().Reset()

	// Somebody who declined to give a name: auto-enrolled, anon_ label.
	// The uuid is a perfectly good key to remember the conversation under.
	snap := &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "anon_73d0a4", UUID: "a1", Area: 3000, TrackID: 54},
	}}

	Speaker().mu.Lock()
	Speaker().latest = &SpeakerResult{TrackID: 54, Name: "anon_73d0a4", UUID: "a1", ResolvedAt: time.Now()}
	Speaker().mu.Unlock()

	uuid, name, measured := snap.AttributedUser()
	require.Equal(t, "a1", uuid, "no name must not mean no memory")
	require.Equal(t, "", name, "anon labels are not names; the greeting falls back to a generic hello")
	require.True(t, measured)
}

func TestAttributedUser_FallsBackToProximity(t *testing.T) {
	Speaker().Reset()
	defer Speaker().Reset()

	snap := twoFaces()
	snap.ClosestUUID, snap.ClosestName = "j1", "jan"

	uuid, name, measured := snap.AttributedUser()
	require.Equal(t, "j1", uuid)
	require.Equal(t, "jan", name)
	require.False(t, measured, "callers must be able to tell a guess from a measurement")
}

func TestAttributedUser_SpeakerOffScreenKeepsIdentity(t *testing.T) {
	Speaker().Reset()
	defer Speaker().Reset()

	snap := twoFaces()
	snap.ClosestUUID, snap.ClosestName = "j1", "jan"

	// Resolved to a track no longer in the snapshot: still theirs, not jan's.
	Speaker().mu.Lock()
	Speaker().latest = &SpeakerResult{TrackID: 99, Name: "wendy", UUID: "w1", ResolvedAt: time.Now()}
	Speaker().mu.Unlock()

	uuid, _, measured := snap.AttributedUser()
	require.Equal(t, "w1", uuid)
	require.True(t, measured)
}

// --- enrolment blocked --------------------------------------------------

func TestToText_SpeakerCannotBeEnrolled(t *testing.T) {
	// Heard clearly, no identity, and standing where the enrolment gate will
	// never clear. The model needs to know the difference between this and
	// "not enrolled yet", because only one of them is worth a word.
	snap := &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "wendy", UUID: "w1", Area: 9000, TrackID: 54},
		{Name: "unknown", UUID: "", Area: 900, TrackID: 77, Enrolling: false},
	}}
	spk := &SpeakerResult{TrackID: 77, Name: "unknown", UUID: "", Score: 0.81, ResolvedAt: timeNow()}

	text := snap.toTextWithSpeaker(spk, true)

	require.Contains(t, text, "cannot be enrolled from where they are")
	require.Contains(t, text, "invite them once")
	require.Contains(t, text, "if they do not, let it go")
}

func TestToText_SpeakerIsBeingEnrolled(t *testing.T) {
	// Same person, but auto-enrol is accumulating samples: naming them is
	// about to become possible, so there is nothing to ask for.
	snap := &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "unknown", UUID: "", Area: 4000, TrackID: 77, Enrolling: true},
	}}
	spk := &SpeakerResult{TrackID: 77, UUID: "", Score: 0.81, ResolvedAt: timeNow()}

	text := snap.toTextWithSpeaker(spk, true)

	require.NotContains(t, text, "cannot be enrolled")
	require.Contains(t, text, "attribute what was just said to THIS person")
}

func TestToText_EnrolledSpeakerNeverAskedToMove(t *testing.T) {
	snap := &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "wendy", UUID: "w1", Area: 3000, TrackID: 54, Enrolling: false},
	}}
	spk := &SpeakerResult{TrackID: 54, Name: "wendy", UUID: "w1", Score: 0.9, ResolvedAt: timeNow()}

	text := snap.toTextWithSpeaker(spk, true)

	require.NotContains(t, text, "cannot be enrolled",
		"someone already in the gallery has nothing to gain by moving")
}

// The failure an operator sees as "it keeps asking me to step closer at
// somebody the video has already labelled anon_xxxx".
func TestToText_EnrolledSpeakerIsNotAskedToMove(t *testing.T) {
	Speaker().Reset()
	defer Speaker().Reset()

	// The face HAS an identity. /speaking does not report one, because the
	// match was not confident at that instant -- routine for anon faces.
	snap := &PresenceSnapshot{Faces: []FaceEntry{
		{Name: "anon_73d0a4", UUID: "a1", Area: 4000, TrackID: 77, Enrolling: false},
	}}
	spk := &SpeakerResult{TrackID: 77, Name: "anon_73d0a4", UUID: "", Score: 0.8, ResolvedAt: timeNow()}
	Speaker().SetLatestForTest(spk)

	text := snap.toTextWithSpeaker(spk, true)

	require.NotContains(t, text, "cannot be enrolled",
		"they already have an identity; moving would change nothing")
	require.NotContains(t, text, "come closer")
	require.Contains(t, text, "attribute what was just said to THIS person")

	// And the rename path must now see that identity.
	require.Equal(t, "a1", Speaker().Latest().UUID,
		"the presence snapshot's uuid should reach the rename path")
}
