package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func speakingServer(t *testing.T, body map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/speaking" {
			http.NotFound(w, r)
			return
		}
		var got map[string]any
		_ = json.NewDecoder(r.Body).Decode(&got)
		if body["_echo_window"] == true {
			body["_got"] = got
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
}

func newTestSpeaker(baseURL string) *SpeakerProvider {
	p := NewSpeakerProvider(baseURL)
	p.Enable()
	return p
}

func TestResolveIdentifiesSpeaker(t *testing.T) {
	uuid := "7fc36a8ae5654cfb994e758436f95e09"
	srv := speakingServer(t, map[string]any{
		"speaker": map[string]any{
			"track_id": 54, "name": "wendy", "uuid": uuid, "score": 0.87,
		},
		"faces": []any{
			map[string]any{"track_id": 54, "name": "wendy", "uuid": uuid, "score": 0.87, "speaking": true, "area": 6390},
			map[string]any{"track_id": 61, "name": "jan", "uuid": nil, "score": 0.10, "speaking": false, "area": 8000},
		},
	})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	start := time.Now().Add(-2 * time.Second)
	got, err := p.Resolve(context.Background(), start, time.Now())
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if !got.Identified() || got.TrackID != 54 {
		t.Fatalf("want track 54, got %+v", got)
	}
	if got.Name != "wendy" || got.UUID != uuid {
		t.Errorf("identity not carried through: %+v", got)
	}
	if len(got.Faces) != 2 || got.Faces[1].Speaking {
		t.Errorf("non-speaker should not be flagged: %+v", got.Faces)
	}
	if p.Latest() == nil {
		t.Error("Latest should return the fresh result")
	}
}

func TestLatestExpires(t *testing.T) {
	srv := speakingServer(t, map[string]any{
		"speaker": map[string]any{"track_id": 1, "name": "x", "score": 0.9},
	})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	p.defaultTTL = 50 * time.Millisecond
	if _, err := p.Resolve(context.Background(), time.Time{}, time.Now()); err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if p.Latest() == nil {
		t.Fatal("should be fresh immediately")
	}
	time.Sleep(80 * time.Millisecond)
	if p.Latest() != nil {
		t.Error("a stale speaker must not be returned; it would be credited with the next utterance")
	}
}

func TestNobodySpeaking(t *testing.T) {
	srv := speakingServer(t, map[string]any{
		"speaker": nil,
		"faces":   []any{map[string]any{"track_id": 3, "name": "wendy", "score": 0.02, "speaking": false}},
	})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	got, err := p.Resolve(context.Background(), time.Now().Add(-time.Second), time.Now())
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if got.Identified() {
		t.Errorf("nobody scored above threshold; must not name anyone: %+v", got)
	}
}

func TestVVADDisabledLatchesOff(t *testing.T) {
	srv := speakingServer(t, map[string]any{"error": "vvad_disabled"})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	if _, err := p.Resolve(context.Background(), time.Now(), time.Now()); err == nil {
		t.Fatal("expected an error")
	}
	if p.Available() {
		t.Error("vvad_disabled is structural; the provider should stop asking")
	}
}

func TestWindowIsSentAndClamped(t *testing.T) {
	srv := speakingServer(t, map[string]any{
		"_echo_window": true,
		"speaker":      map[string]any{"track_id": 7, "name": "a", "score": 0.9},
	})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	end := time.Now()
	got, err := p.Resolve(context.Background(), end.Add(-90*time.Second), end)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	span := got.WindowEnd.Sub(got.WindowStart).Seconds()
	if span > speakerMaxWindowSec+0.5 {
		t.Errorf("window not clamped: %.1fs", span)
	}
}

func TestResolveAsyncInvalidatesPreviousSpeaker(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release // hold the second resolve open
		_ = json.NewEncoder(w).Encode(map[string]any{
			"speaker": map[string]any{"track_id": 61, "name": "jan", "score": 0.9},
		})
	}))
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	// Seed the previous utterance's answer.
	p.mu.Lock()
	p.latest = &SpeakerResult{TrackID: 54, Name: "wendy", ResolvedAt: time.Now()}
	p.mu.Unlock()
	require.NotNil(t, p.Latest())

	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())

	require.Nil(t, p.Latest(), "stale speaker must not survive into the next utterance")
	require.True(t, p.Pending())

	close(release)
	p.WaitFresh(context.Background())
	got := p.Latest()
	require.NotNil(t, got)
	require.Equal(t, 61, got.TrackID, "the new utterance's speaker, not the old one")
}

func TestWaitFreshReturnsWhenResolved(t *testing.T) {
	srv := speakingServer(t, map[string]any{
		"speaker": map[string]any{"track_id": 7, "name": "a", "score": 0.9},
	})
	defer srv.Close()

	p := newTestSpeaker(srv.URL)
	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())

	start := time.Now()
	p.WaitFresh(context.Background())
	require.Less(t, time.Since(start), 2*time.Second, "should return on completion, not on timeout")
	require.NotNil(t, p.Latest(), "the answer is available to the prompt being built")
	require.False(t, p.Pending())
}

func TestWaitFreshNoOpWhenNothingPending(t *testing.T) {
	p := newTestSpeaker("http://127.0.0.1:1")
	start := time.Now()
	p.WaitFresh(context.Background())
	require.Less(t, time.Since(start), 50*time.Millisecond)
}

// The verdict describes one utterance, so its shelf life should come from
// that utterance. A fixed constant treats a half-second "yeah" and a
// five-second introduction as equally durable, and the constant was picked
// by hand rather than measured.

func TestTTLScalesWithHowLongTheySpoke(t *testing.T) {
	p := newTestSpeaker("")
	end := time.Now()

	mk := func(d time.Duration) *SpeakerResult {
		return &SpeakerResult{WindowStart: end.Add(-d), WindowEnd: end}
	}

	require.Equal(t, speakerTTLMin, p.resultTTL(mk(200*time.Millisecond)),
		"a clipped one-word reply still needs long enough to reach the prompt")
	require.Equal(t, 3*time.Second, p.resultTTL(mk(time.Second)))
	require.Equal(t, 6*time.Second, p.resultTTL(mk(2*time.Second)))
	require.Equal(t, speakerTTLMax, p.resultTTL(mk(30*time.Second)),
		"a runaway window must not grant the speaker slot for most of a minute")
}

func TestTTLFallsBackWithoutAWindow(t *testing.T) {
	p := newTestSpeaker("")
	require.Equal(t, p.defaultTTL, p.resultTTL(&SpeakerResult{}))
	require.Equal(t, p.defaultTTL, p.resultTTL(nil))
}

func TestLatestExpiresAgainstItsOwnUtterance(t *testing.T) {
	p := newTestSpeaker("")
	now := time.Now()

	p.mu.Lock()
	p.latest = &SpeakerResult{
		TrackID: 1, ResolvedAt: now.Add(-3 * time.Second),
		WindowStart: now.Add(-3300 * time.Millisecond), WindowEnd: now.Add(-3 * time.Second),
	}
	p.mu.Unlock()
	require.Nil(t, p.Latest(), "a very short utterance should not linger")

	p.mu.Lock()
	p.latest = &SpeakerResult{
		TrackID: 1, ResolvedAt: now.Add(-3 * time.Second),
		WindowStart: now.Add(-7 * time.Second), WindowEnd: now.Add(-3 * time.Second),
	}
	p.mu.Unlock()
	require.NotNil(t, p.Latest(), "a long introduction stays relevant longer")
}

func TestWaitFreshEndsWhenTheContextDoes(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	defer srv.Close()
	defer close(block)

	p := newTestSpeaker(srv.URL)
	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	p.WaitFresh(ctx)
	require.Less(t, time.Since(start), time.Second,
		"a cancelled tick must not sit on a wedged endpoint")
}

func TestATurnProceedsWithoutAnAnswer(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	defer srv.Close()
	defer close(block)

	p := newTestSpeaker(srv.URL)
	p.maxWait = 50 * time.Millisecond
	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())

	start := time.Now()
	p.WaitFresh(context.Background())
	elapsed := time.Since(start)

	require.Less(t, elapsed, time.Second,
		"the reply must not wait on the network timeout")
	require.GreaterOrEqual(t, elapsed, 40*time.Millisecond)
	require.Nil(t, p.Latest(), "and it proceeds with an honest absence")
}

func TestDetectionIsOffUntilEnabled(t *testing.T) {
	asked := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked <- struct{}{}
		<-r.Context().Done()
	}))
	defer srv.Close()

	p := NewSpeakerProvider(srv.URL)
	require.False(t, p.Available(), "speaker detection is opt-in")

	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())
	require.False(t, p.Pending(), "nothing should be dispatched")

	start := time.Now()
	p.WaitFresh(context.Background())
	require.Less(t, time.Since(start), 20*time.Millisecond,
		"a feature that is off must not delay the transcript at all")

	select {
	case <-asked:
		t.Fatal("the service must not be contacted while detection is off")
	default:
	}

	p.Enable()
	require.True(t, p.Available())
	p.ResolveAsync(time.Now().Add(-time.Second), time.Now())
	require.True(t, p.Pending())
	<-asked
}
