package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	speakerDefaultBaseURL = "http://127.0.0.1:6793"
	speakerDefaultTimeout = 2 * time.Second
	speakerDefaultTTL     = 5 * time.Second
	speakerTTLPerSecond   = 3.0
	speakerTTLMin         = 2 * time.Second
	speakerTTLMax         = 12 * time.Second
	speakerMaxWindowSec   = 15.0
	speakerWaitDefault    = 300 * time.Millisecond
	speakerWaitEnv        = "OM1_SPEAKER_WAIT_MS"
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
	TrackID int
	Name    string
	UUID    string
	Score   float64
	Faces   []SpeakerFace

	ResolvedAt time.Time

	WindowStart time.Time
	WindowEnd   time.Time
}

// identityUUID returns the speaker's UUID, or "" if none.
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

	enabled  bool
	latest   *SpeakerResult
	pending  chan struct{}
	disabled bool
	lastErr  error

	mu sync.RWMutex
}

var (
	speakerWaitOnce  sync.Once
	speakerWaitValue time.Duration

	speakerMu       sync.RWMutex
	speakerInstance *SpeakerProvider
)

func Speaker() *SpeakerProvider {
	speakerMu.RLock()
	p := speakerInstance
	speakerMu.RUnlock()
	if p != nil {
		return p
	}

	speakerMu.Lock()
	defer speakerMu.Unlock()
	if speakerInstance == nil {
		speakerInstance = NewSpeakerProvider("")
	}
	return speakerInstance
}

func SetSpeaker(p *SpeakerProvider) {
	if p == nil {
		p = NewSpeakerProvider("")
	}
	speakerMu.Lock()
	speakerInstance = p
	speakerMu.Unlock()
}

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

func (p *SpeakerProvider) Enable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabled = true
}

func (p *SpeakerProvider) Available() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.enabled && !p.disabled
}

func (p *SpeakerProvider) ResolveAsync(start, end time.Time) {
	if !p.Available() {
		return
	}

	done := make(chan struct{})
	p.mu.Lock()
	if p.pending != nil {
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

func (p *SpeakerProvider) WaitFresh(ctx context.Context) {
	p.mu.RLock()
	done := p.pending
	off := !p.enabled || p.disabled
	p.mu.RUnlock()
	if done == nil || off {
		return
	}

	timer := time.NewTimer(speakerWait())
	defer timer.Stop()

	select {
	case <-done:
	case <-timer.C:
	case <-ctx.Done():
	}
}

func speakerWait() time.Duration {
	speakerWaitOnce.Do(func() {
		speakerWaitValue = speakerWaitDefault
		raw := os.Getenv(speakerWaitEnv)
		if raw == "" {
			return
		}
		ms, err := strconv.Atoi(raw)
		if err != nil || ms <= 0 {
			return
		}
		speakerWaitValue = time.Duration(ms) * time.Millisecond
	})
	return speakerWaitValue
}

func (p *SpeakerProvider) Pending() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.pending != nil
}

func (p *SpeakerProvider) Resolve(ctx context.Context, start, end time.Time) (*SpeakerResult, error) {
	if end.IsZero() {
		end = time.Now()
	}

	body := map[string]any{}
	if !start.IsZero() && end.After(start) {
		if end.Sub(start).Seconds() > speakerMaxWindowSec {
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

func (p *SpeakerProvider) noteErr(err error) {
	p.mu.Lock()
	p.lastErr = err
	p.mu.Unlock()
}
