package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

// Who was talking, resolved from the audio-visual model rather than guessed.
//
// The presence line used to answer "who is addressing the robot" with the
// largest face box, on the reasoning that the nearest person is probably the
// one speaking. With one person in front of the robot that is right often
// enough to look like it works. With two it is a coin toss decided by who
// leaned in, and the robot attributes what it just heard -- including
// somebody's name -- to the wrong face.
//
// The video-processor already answers the question properly: LR-ASD scores
// each visible face against the audio that overlaps it, so somebody chewing
// beside the speaker scores low and the speaker scores high. It exposes that
// on /speaking, and the endpoint is built to be asked about ONE UTTERANCE --
// "who was talking between these two timestamps" -- not about the current
// instant. That is why this provider is driven by the ASR's own speech
// boundaries instead of polling: a poll would average over whatever happened
// to be in the window, and the answer needs to be about the sentence the LLM
// is about to read.
//
// Degrades honestly. If the endpoint is missing, disabled (no --lrasd-engine,
// which reports "vvad_disabled"), or nobody scores above the threshold, the
// result is simply absent and the caller says so rather than substituting a
// guess dressed up as a measurement.

const (
	speakerDefaultBaseURL = "http://127.0.0.1:6793"
	speakerDefaultTimeout = 2 * time.Second

	// How long a resolved speaker stays usable, when the utterance it came
	// from is not known.
	//
	// Only the lookback path lands here. Everything driven by the ASR carries
	// a real window and expires against that instead -- see resultTTL.
	speakerDefaultTTL = 5 * time.Second

	// Shelf life as a multiple of how long the person actually spoke.
	//
	// The answer is a statement about one utterance, so how long it stays
	// worth showing should come from that utterance rather than from a
	// constant somebody picked. A three-second sentence establishes who is
	// talking to the robot; a half-second "yeah" barely does, and the two
	// should not linger equally.
	speakerTTLPerSecond = 3.0

	// Floor and ceiling on the above. The floor keeps a clipped one-word
	// reply usable long enough to reach the prompt at all; the ceiling stops
	// a long monologue -- or a window stretched by a slow transcript --
	// granting somebody the speaker slot for most of a minute.
	speakerTTLMin = 2 * time.Second
	speakerTTLMax = 12 * time.Second

	// Guard against an utterance window so long that it spans several
	// speakers. /speaking scores the whole span, so a runaway window
	// returns whoever dominated it rather than who finished it.
	speakerMaxWindowSec = 15.0
)

// SpeakerFace is one face's verdict for an utterance window.
type SpeakerFace struct {
	TrackID  int     `json:"track_id"`
	Name     string  `json:"name"`
	UUID     *string `json:"uuid"`
	Score    float64 `json:"score"`
	Speaking bool    `json:"speaking"`
	Area     int     `json:"area"`
}

// speakerIdentity is the chosen speaker in a /speaking response.
type speakerIdentity struct {
	TrackID int     `json:"track_id"`
	Name    string  `json:"name"`
	UUID    *string `json:"uuid"`
	Score   float64 `json:"score"`
}

// speakingResponse is the parsed /speaking body.
type speakingResponse struct {
	Speaker *speakerIdentity `json:"speaker"`
	Faces   []SpeakerFace    `json:"faces"`
	Error   string           `json:"error"`
}

// SpeakerResult is one resolved utterance: who said it, and how the rest scored.
type SpeakerResult struct {
	// TrackID is the face the audio-visual model picked. -1 when nobody
	// scored above the threshold -- silence, an off-camera voice, or the
	// robot hearing itself.
	TrackID int
	Name    string
	UUID    string
	Score   float64
	Faces   []SpeakerFace

	// ResolvedAt is when the answer arrived, for the staleness check.
	ResolvedAt time.Time
	// Window is the utterance span that was scored.
	WindowStart time.Time
	WindowEnd   time.Time
}

// identityUUID is the speaker verdict's own copy of the identity, used only
// as a fallback. It is null whenever the match was not confident at that
// instant, which is routine for an auto-enrolled face, so the face entry in
// the presence snapshot is the better source.
func (r *SpeakerResult) identityUUID() string {
	if r == nil {
		return ""
	}
	return r.UUID
}

// Identified reports whether a specific face was picked out.
func (r *SpeakerResult) Identified() bool { return r != nil && r.TrackID >= 0 }

// SpeakerProvider resolves utterance windows to faces via /speaking.
type SpeakerProvider struct {
	baseURL string
	client  *http.Client
	ttl     time.Duration

	mu     sync.RWMutex
	latest *SpeakerResult
	// pending is closed when the in-flight resolve finishes. Non-nil only
	// while one is running.
	//
	// This exists because the transcript and the prompt are the same event:
	// an accepted transcript fires the input orchestrator's TickNow, and the
	// cortex loop builds the prompt straight away. An HTTP round trip cannot
	// finish inside that window, so without somewhere to wait, the prompt
	// carries the PREVIOUS utterance's speaker -- and two people taking
	// turns get named as each other, every turn. That is worse than not
	// knowing: it is confidently wrong, which is the exact failure the
	// measurement was introduced to remove.
	pending chan struct{}
	// disabled latches when the endpoint reports vvad_disabled, so a rig
	// running without the LR-ASD engine does not pay for a request per
	// utterance forever. Cleared by Reset.
	disabled bool
	// lastErr is the most recent failure, surfaced for diagnostics rather
	// than swallowed -- "the LLM does not know who is talking" is a fault
	// worth being able to see.
	lastErr error
}

var (
	speakerOnce     sync.Once
	speakerInstance *SpeakerProvider
)

// Speaker returns the singleton SpeakerProvider.
func Speaker() *SpeakerProvider {
	speakerOnce.Do(func() { speakerInstance = NewSpeakerProvider("") })
	return speakerInstance
}

// NewSpeakerProvider constructs a provider. An empty baseURL uses the default.
func NewSpeakerProvider(baseURL string) *SpeakerProvider {
	if baseURL == "" {
		baseURL = speakerDefaultBaseURL
	}
	return &SpeakerProvider{
		baseURL: baseURL,
		ttl:     speakerDefaultTTL,
		client: &http.Client{
			Transport: httpclient.Default().Transport,
			Timeout:   speakerDefaultTimeout,
		},
	}
}

// Latest returns the most recent resolution, or nil when there is none or it
// has aged past the TTL.
//
// Expiry is the point, not a detail. A stale answer looks exactly like a fresh
// one to whoever renders it, so a speaker held past their utterance would keep
// being credited with sentences somebody else said.
func (p *SpeakerProvider) Latest() *SpeakerResult {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.latest == nil {
		return nil
	}
	if time.Since(p.latest.ResolvedAt) > p.resultTTL(p.latest) {
		return nil
	}
	cp := *p.latest
	return &cp
}

// resultTTL is how long this particular answer stays usable.
//
// Derived from the utterance rather than fixed, because that is what the
// answer is about: it says who was speaking between two timestamps, and how
// long that remains the useful answer to "who is the robot talking to" scales
// with how much was said. p.ttl is the fallback for a result with no window,
// which is only the lookback path.
func (p *SpeakerProvider) resultTTL(r *SpeakerResult) time.Duration {
	if r == nil || r.WindowStart.IsZero() || !r.WindowEnd.After(r.WindowStart) {
		return p.ttl
	}
	spoken := r.WindowEnd.Sub(r.WindowStart)
	ttl := time.Duration(float64(spoken) * speakerTTLPerSecond)
	if ttl < speakerTTLMin {
		return speakerTTLMin
	}
	if ttl > speakerTTLMax {
		return speakerTTLMax
	}
	return ttl
}

// Available reports whether the endpoint is answering. False once it has said
// vvad_disabled, which means the pipeline is running without an LR-ASD engine.
func (p *SpeakerProvider) Available() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return !p.disabled
}

// LastError returns the most recent failure, or nil.
func (p *SpeakerProvider) LastError() error {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lastErr
}

// Reset clears the cached answer and the disabled latch. Called when a mode
// changes, so a conversation never opens holding the previous one's speaker.
func (p *SpeakerProvider) Reset() {
	p.mu.Lock()
	if p.pending != nil {
		close(p.pending)
		p.pending = nil
	}
	p.latest = nil
	p.disabled = false
	p.lastErr = nil
	p.mu.Unlock()
}

// ResolveAsync resolves an utterance window in the background.
//
// Fire-and-forget on purpose: this sits on the ASR's transcript path, which
// must not wait on an HTTP round trip to hand a sentence to the LLM. The
// answer lands before the model is prompted in the normal case, and when it
// does not, Latest simply returns nil and the presence line says the speaker
// is unknown -- which is true, and better than delaying every utterance for
// the one that arrives late.
func (p *SpeakerProvider) ResolveAsync(start, end time.Time) {
	if !p.Available() {
		return
	}

	// Drop the previous answer BEFORE the new one is asked for. It describes
	// the utterance before this one, and leaving it in place is what turns a
	// late reply into a wrong attribution rather than an absent one. From
	// here until the resolve lands, Latest reports nothing and the prompt
	// says the speaker is still being worked out -- which is true.
	done := make(chan struct{})
	p.mu.Lock()
	if p.pending != nil {
		// A resolve is already running for an earlier utterance. Retire it:
		// whoever is waiting wants THIS one.
		close(p.pending)
	}
	p.latest = nil
	p.pending = done
	p.mu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), speakerDefaultTimeout+time.Second)
		defer cancel()
		_, _ = p.Resolve(ctx, start, end)

		p.mu.Lock()
		if p.pending == done {
			p.pending = nil
		}
		p.mu.Unlock()
		close(done)
	}()
}

// WaitFresh blocks until the in-flight resolve finishes, or timeout elapses,
// or ctx is cancelled. Returns immediately when nothing is pending.
//
// Called by the consumer rather than the producer on purpose. Blocking the
// ASR websocket read loop to wait for this would stall transcript delivery
// for every utterance including the ones nobody asks about; blocking the
// tick costs the same milliseconds only where the answer is about to be
// used, and the tick is already about to make a much slower LLM call.
func (p *SpeakerProvider) WaitFresh(ctx context.Context, timeout time.Duration) {
	p.mu.RLock()
	done := p.pending
	p.mu.RUnlock()
	if done == nil {
		return
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
	case <-ctx.Done():
	}
}

// Pending reports whether a resolve is in flight.
func (p *SpeakerProvider) Pending() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.pending != nil
}

// Resolve asks /speaking who was talking between start and end.
//
// A zero start falls back to the endpoint's own lookback mode, which scores
// the last few seconds. That is the weaker question -- it cannot know where
// the sentence began -- but it is the only one available when the vendor
// protocol gave no speech-start event.
func (p *SpeakerProvider) Resolve(ctx context.Context, start, end time.Time) (*SpeakerResult, error) {
	if end.IsZero() {
		end = time.Now()
	}

	body := map[string]any{}
	if !start.IsZero() && end.After(start) {
		if end.Sub(start).Seconds() > speakerMaxWindowSec {
			// Keep the tail: the end of a long window is the part that
			// produced the transcript just accepted.
			start = end.Add(-time.Duration(speakerMaxWindowSec * float64(time.Second)))
		}
		body["win_start_ms"] = start.UnixMilli()
		body["win_end_ms"] = end.UnixMilli()
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+"/speaking", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		p.noteErr(err)
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("speaking: status %d", resp.StatusCode)
		p.noteErr(err)
		return nil, err
	}

	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		p.noteErr(err)
		return nil, err
	}

	var parsed speakingResponse
	if err := json.Unmarshal(payload, &parsed); err != nil {
		p.noteErr(err)
		return nil, err
	}

	if parsed.Error != "" {
		err := fmt.Errorf("speaking: %s", parsed.Error)
		if parsed.Error == "vvad_disabled" {
			// Structural, not transient: the pipeline was started without
			// an LR-ASD engine and no amount of retrying will change that.
			p.mu.Lock()
			p.disabled = true
			p.lastErr = err
			p.mu.Unlock()
			return nil, err
		}
		p.noteErr(err)
		return nil, err
	}

	result := &SpeakerResult{
		TrackID:     -1,
		Faces:       parsed.Faces,
		ResolvedAt:  time.Now(),
		WindowStart: start,
		WindowEnd:   end,
	}
	if parsed.Speaker != nil {
		result.TrackID = parsed.Speaker.TrackID
		result.Name = parsed.Speaker.Name
		result.Score = parsed.Speaker.Score
		if parsed.Speaker.UUID != nil {
			result.UUID = *parsed.Speaker.UUID
		}
	}

	p.mu.Lock()
	p.latest = result
	p.lastErr = nil
	p.mu.Unlock()

	cp := *result
	return &cp, nil
}

// NoteIdentity records the UUID the presence snapshot holds for a track.
//
// The speaker verdict carries its own copy, and it is null whenever the match
// was not confident at that instant -- routine for an auto-enrolled face. The
// presence snapshot resolves the same track to the identity the video is
// actually labelling, so whoever renders that line hands the answer back here
// and every later consumer, the rename included, sees the identity rather
// than a hole where one should be.
//
// Ignored when it does not describe the current speaker: a late call about a
// previous utterance must not overwrite this one's identity.
func (p *SpeakerProvider) NoteIdentity(trackID int, uuid string) {
	if uuid == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.latest == nil || p.latest.TrackID != trackID {
		return
	}
	p.latest.UUID = uuid
}

// SetLatestForTest installs a resolution directly. Test-only: the real path
// goes through Resolve so that the window, the staleness stamp and the
// pending latch all stay consistent with each other.
func (p *SpeakerProvider) SetLatestForTest(r *SpeakerResult) {
	p.mu.Lock()
	p.latest = r
	p.mu.Unlock()
}

// noteErr records a transient failure without latching the provider off.
func (p *SpeakerProvider) noteErr(err error) {
	p.mu.Lock()
	p.lastErr = err
	p.mu.Unlock()
}
