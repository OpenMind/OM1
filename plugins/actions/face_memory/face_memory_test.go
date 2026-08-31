package face_memory

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

// captured is one request the fake video-processor received.
type captured struct {
	path string
	body map[string]any
}

func fakeVideoProcessor(t *testing.T, reply map[string]any) (*httptest.Server, *[]captured, *sync.Mutex) {
	t.Helper()
	var mu sync.Mutex
	var got []captured
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		got = append(got, captured{path: r.URL.Path, body: body})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(reply)
	}))
	return srv, &got, &mu
}

func testConnector(t *testing.T, baseURL string) *Connector {
	t.Helper()
	return &Connector{
		log:    zap.NewNop(),
		cfg:    Config{FaceHTTPBaseURL: baseURL, HTTPTimeoutSec: 2},
		client: &http.Client{Timeout: 2 * time.Second},
	}
}

func setSpeaker(trackID int, name, uuid string) {
	providers.Speaker().SetLatestForTest(nil)
	providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
		TrackID: trackID, Name: name, UUID: uuid, ResolvedAt: time.Now(),
	})
}

// The scenario this exists for: the person being talked to is in FRONT and has
// the largest face; somebody behind them says "my name is Sean". Renaming the
// largest face overwrites the wrong identity, and does it silently.
func TestSetNameTargetsTheSpeakerNotTheNearestFace(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "s1", "name": "sean", "created": true, "sample_count": 3,
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(77, "unknown", "") // Sean: heard, but not enrolled yet
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "Sean"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1)
	req := (*got)[0]

	require.Equal(t, "/set_name", req.path,
		"/set_name_current renames the largest face, which is not who spoke")
	require.EqualValues(t, 77, req.body["track_id"],
		"the speaker is addressed by track, so an unenrolled person can still be named")
	require.Equal(t, "sean", req.body["name"])
	require.NotContains(t, req.body, "uuid",
		"uuid would only work for someone already enrolled; track_id covers both cases")
}

func TestSetNameFallsBackWhenNoSpeakerMeasured(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "w1", "name": "wendy",
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	providers.Speaker().SetLatestForTest(nil) // nothing resolved
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "wendy"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1)
	require.Equal(t, "/set_name_current", (*got)[0].path,
		"a rig with no LR-ASD engine must still be able to learn names")
}

func TestSetNameFallsBackWhenNobodyScoredAsSpeaking(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{"ok": true, "name": "x"})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	// Resolved, but nobody cleared the threshold: TrackID -1 is not an identity.
	providers.Speaker().SetLatestForTest(nil)
	providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
		TrackID: -1, ResolvedAt: time.Now(),
	})
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "x"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, "/set_name_current", (*got)[0].path)
}

func TestSetNameRejectsEmptyName(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{"ok": true})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(77, "", "")
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "  "})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Empty(t, *got, "no name means no request; nothing should be renamed")
}

// The failure the operator actually saw: someone at the back says "my name is
// Sean" and the name lands on the enrolled person standing in front.
func TestSetNameRefusesWhenSeveralFacesAreComparable(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{"ok": true, "name": "sean"})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	// Two people side by side: nothing distinguishes them by area, and the
	// audio model had no opinion.
	providers.Speaker().SetLatestForTest(nil)
	providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
		TrackID:    -1,
		ResolvedAt: time.Now(),
		Faces: []providers.SpeakerFace{
			{TrackID: 54, Name: "wendy", Area: 9000},
			{TrackID: 77, Name: "unknown", Area: 8200},
		},
	})

	c := testConnector(t, srv.URL)
	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "Sean"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Empty(t, *got,
		"overwriting a correct identity is worse than declining to record a name")
}

// Someone standing at the robot with the next person well behind them. The
// audio model abstained, but the scene is not actually ambiguous, and
// refusing here made the feature unusable whenever a bystander was in frame.
func TestSetNameAcceptsAClearlyDominantFace(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "s1", "name": "sean",
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	uuid := "s1"
	providers.Speaker().SetLatestForTest(nil)
	providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
		TrackID:    -1,
		ResolvedAt: time.Now(),
		Faces: []providers.SpeakerFace{
			{TrackID: 77, Name: "unknown", UUID: &uuid, Area: 9000}, // ~1.5 m
			{TrackID: 54, Name: "wendy", Area: 3000},                // ~2.6 m
		},
	})

	c := testConnector(t, srv.URL)
	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "Sean"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1)
	require.Equal(t, "/set_name", (*got)[0].path)
	require.Equal(t, "s1", (*got)[0].body["uuid"], "the dominant face, addressed by identity")
}

func TestDominanceRatioIsTheBoundary(t *testing.T) {
	mk := func(a, b int) int {
		providers.Speaker().SetLatestForTest(nil)
		providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
			TrackID: -1, ResolvedAt: time.Now(),
			Faces: []providers.SpeakerFace{
				{TrackID: 1, Area: a}, {TrackID: 2, Area: b},
			},
		})
		d, n := dominantFace()
		require.Equal(t, 2, n)
		if d == nil {
			return 0
		}
		return d.TrackID
	}
	defer providers.Speaker().SetLatestForTest(nil)

	require.Equal(t, 0, mk(2400, 1000), "2.4x is not clear enough")
	require.Equal(t, 1, mk(2500, 1000), "2.5x is the boundary")
	require.Equal(t, 1, mk(9000, 1000), "plainly dominant")
	require.Equal(t, 0, mk(1000, 1000), "identical")
}

func TestSetNameStillWorksWithASingleFace(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "w1", "name": "wendy",
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	providers.Speaker().SetLatestForTest(nil)
	providers.Speaker().SetLatestForTest(&providers.SpeakerResult{
		TrackID:    -1,
		ResolvedAt: time.Now(),
		Faces:      []providers.SpeakerFace{{TrackID: 54, Name: "wendy", Area: 9000}},
	})

	c := testConnector(t, srv.URL)
	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "wendy"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1, "with one face there is nothing to get wrong")
	require.Equal(t, "/set_name_current", (*got)[0].path)
}

// Somebody the video already labels anon_xxxx HAS an identity. Renaming them
// must not depend on the receiver re-deriving it from a track id.
func TestSetNameUsesTheSpeakerUUIDWhenKnown(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "s1", "name": "sean",
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(77, "anon_73d0a4", "s1") // enrolled, unnamed
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "Sean"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1)
	require.Equal(t, "/set_name", (*got)[0].path)
	require.Equal(t, "s1", (*got)[0].body["uuid"], "the identity we already hold")
	require.NotContains(t, (*got)[0].body, "track_id",
		"a track lookup can miss a face the video has plainly identified")
}

// A UUID can disappear between the utterance and the rename. The person is
// still standing there, so fall back to the track rather than losing a name
// they just said out loud.
func TestSetNameRetriesByTrackWhenUUIDVanished(t *testing.T) {
	var mu sync.Mutex
	var got []captured
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		mu.Lock()
		got = append(got, captured{path: r.URL.Path, body: body})
		n := len(got)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "uuid_not_found"})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "uuid": "new", "name": "sean"})
	}))
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(77, "anon_73d0a4", "gone")
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(), map[string]any{"to_id": "Sean"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, got, 2, "should retry once")
	require.Equal(t, "gone", got[0].body["uuid"])
	require.EqualValues(t, 77, got[1].body["track_id"])
}

// The exact failure from the field: the model passed the name through as it
// was spoken and the receiver rejected it on a character class.
func TestNormalizeIDMatchesReceiverRules(t *testing.T) {
	valid := regexp.MustCompile(`^[a-z0-9_-]{1,32}$`)

	cases := map[string]string{
		"Li Fan":                "li-fan", // the one that failed: bad_id invalid_chars
		"wendy":                 "wendy",
		"  Sean  ":              "sean",
		"Jerin Peter":           "jerin-peter",
		"O'Brien":               "o-brien",
		"Anne-Marie":            "anne-marie",
		"J. R. Smith":           "j-r-smith",
		"wendy_2":               "wendy_2",
		"Li  Fan":               "li-fan", // repeated separators collapse
		"Zoë":                   "zo",     // accents dropped, not guessed at
		strings.Repeat("a", 40): strings.Repeat("a", 32),
	}

	for in, want := range cases {
		got := normalizeID(in)
		if got != want {
			t.Errorf("normalizeID(%q) = %q, want %q", in, got, want)
		}
		if got != "" && !valid.MatchString(got) {
			t.Errorf("normalizeID(%q) = %q, which the receiver would reject", in, got)
		}
	}
}

func TestNormalizeIDRejectsUnusableNames(t *testing.T) {
	// Nothing usable left: better an empty id, which doSetName refuses with a
	// clear message, than a stray dash the gallery would happily store.
	for _, in := range []string{"", "   ", "---", "你好", "!!!"} {
		if got := normalizeID(in); got != "" {
			t.Errorf("normalizeID(%q) = %q, want empty", in, got)
		}
	}
}

// Observed in the field: op=set_name arriving with the name in `id` rather
// than `to_id`. The action carries a field per operation, so a name has two
// plausible slots and the model picked the other one -- and a name that had
// been heard, transcribed and normalised correctly was dropped as "bad_id".
func TestSetNameAcceptsNameInEitherField(t *testing.T) {
	for _, field := range []string{"to_id", "id"} {
		t.Run(field, func(t *testing.T) {
			srv, got, mu := fakeVideoProcessor(t, map[string]any{
				"ok": true, "uuid": "s1", "name": "li-fan",
			})
			defer srv.Close()
			defer providers.Speaker().SetLatestForTest(nil)

			setSpeaker(77, "anon_1", "s1")
			c := testConnector(t, srv.URL)

			_, err := c.doSetName(context.Background(), map[string]any{field: "Li Fan"})
			require.NoError(t, err)

			mu.Lock()
			defer mu.Unlock()
			require.Len(t, *got, 1, "the name must reach the receiver from either field")
			require.Equal(t, "li-fan", (*got)[0].body["name"])
		})
	}
}

func TestSetNamePrefersToIDWhenBothPresent(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{"ok": true, "name": "b"})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(77, "anon_1", "s1")
	c := testConnector(t, srv.URL)

	_, err := c.doSetName(context.Background(),
		map[string]any{"to_id": "correct", "id": "wrong"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, "correct", (*got)[0].body["name"])
}

// Observed: the model chose op=selfie for someone the video labelled
// "newcomer", /selfie photographed the most prominent face instead, and the
// name came back "face_belongs_to claimed=difan matched=wendy" -- the speaker's
// name rejected because it landed on the person standing in front.
func TestSelfieBecomesRenameWhenTheSpeakerIsAlreadyEnrolled(t *testing.T) {
	srv, got, mu := fakeVideoProcessor(t, map[string]any{
		"ok": true, "uuid": "d1", "name": "difan",
	})
	defer srv.Close()
	defer providers.Speaker().SetLatestForTest(nil)

	setSpeaker(3, "anon_f3b609", "d1") // auto-enrolled, unnamed
	c := testConnector(t, srv.URL)

	_, err := c.doSelfie(context.Background(), map[string]any{"id": "Difan"})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, *got, 1)
	require.Equal(t, "/set_name", (*got)[0].path, "an enrolled face needs a name, not a photo")
	require.Equal(t, "d1", (*got)[0].body["uuid"], "and it must be the speaker's identity")
	require.Equal(t, "difan", (*got)[0].body["name"])
}
