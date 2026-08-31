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

const (
	speakerDefaultBaseURL     = "http://127.0.0.1:6793"
	speakerDefaultHTTPTimeout = 2 * time.Second
	speakerDefaultMaxWait     = 300 * time.Millisecond
	speakerDefaultTTL         = 5 * time.Second
	speakerTTLPerSecond       = 3.0
	speakerTTLMin             = 2 * time.Second
	speakerTTLMax             = 12 * time.Second
	speakerMaxWindowSec       = 15.0
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

	defaultTTL time.Duration
	maxWait    time.Duration

	enabled     bool
	unsupported bool

	latest      *SpeakerResult
	resolveDone chan struct{}
	lastErr     error

	mu sync.RWMutex
}

var (
	speakerInstanceMu sync.RWMutex
	speakerInstance   *SpeakerProvider
)

func Speaker() *SpeakerProvider {
	speakerInstanceMu.RLock()
	provider := speakerInstance
	speakerInstanceMu.RUnlock()
	if provider != nil {
		return provider
	}

	speakerInstanceMu.Lock()
	defer speakerInstanceMu.Unlock()
	if speakerInstance == nil {
		speakerInstance = NewSpeakerProvider("")
	}
	return speakerInstance
}

func SetSpeaker(provider *SpeakerProvider) {
	if provider == nil {
		provider = NewSpeakerProvider("")
	}
	speakerInstanceMu.Lock()
	speakerInstance = provider
	speakerInstanceMu.Unlock()
}

func NewSpeakerProvider(baseURL string) *SpeakerProvider {
	if baseURL == "" {
		baseURL = speakerDefaultBaseURL
	}
	return &SpeakerProvider{
		baseURL:    baseURL,
		defaultTTL: speakerDefaultTTL,
		maxWait:    speakerDefaultMaxWait,
		client: &http.Client{
			Transport: httpclient.Default().Transport,
			Timeout:   speakerDefaultHTTPTimeout,
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
	snapshot := *p.latest
	return &snapshot
}

func (p *SpeakerProvider) resultTTL(result *SpeakerResult) time.Duration {
	if result == nil || result.WindowStart.IsZero() || !result.WindowEnd.After(result.WindowStart) {
		return p.defaultTTL
	}
	spoken := result.WindowEnd.Sub(result.WindowStart)
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
	return p.enabled && !p.unsupported
}

func (p *SpeakerProvider) ResolveAsync(start, end time.Time) {
	if !p.Available() {
		return
	}

	resolved := make(chan struct{})
	p.mu.Lock()
	if p.resolveDone != nil {
		close(p.resolveDone)
	}
	p.latest = nil
	p.resolveDone = resolved
	p.mu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(
			context.Background(), speakerDefaultHTTPTimeout+time.Second)
		defer cancel()
		_, _ = p.Resolve(ctx, start, end)

		p.mu.Lock()
		if p.resolveDone == resolved {
			p.resolveDone = nil
		}
		p.mu.Unlock()
		close(resolved)
	}()
}

func (p *SpeakerProvider) WaitFresh(ctx context.Context) {
	p.mu.RLock()
	resolved := p.resolveDone
	unavailable := !p.enabled || p.unsupported
	maxWait := p.maxWait
	p.mu.RUnlock()
	if resolved == nil || unavailable {
		return
	}

	timer := time.NewTimer(maxWait)
	defer timer.Stop()

	select {
	case <-resolved:
	case <-timer.C:
	case <-ctx.Done():
	}
}

func (p *SpeakerProvider) Pending() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.resolveDone != nil
}

func (p *SpeakerProvider) Resolve(ctx context.Context, start, end time.Time) (*SpeakerResult, error) {
	if end.IsZero() {
		end = time.Now()
	}

	reqFields := map[string]any{}
	if !start.IsZero() && end.After(start) {
		if end.Sub(start).Seconds() > speakerMaxWindowSec {
			start = end.Add(-time.Duration(speakerMaxWindowSec * float64(time.Second)))
		}
		reqFields["win_start_ms"] = start.UnixMilli()
		reqFields["win_end_ms"] = end.UnixMilli()
	}

	reqBody, err := json.Marshal(reqFields)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+"/speaking", bytes.NewReader(reqBody))
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

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		p.noteErr(err)
		return nil, err
	}

	var parsed speakingResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		p.noteErr(err)
		return nil, err
	}

	if parsed.Error != "" {
		err := fmt.Errorf("speaking: %s", parsed.Error)
		if parsed.Error == "vvad_disabled" {
			p.mu.Lock()
			p.unsupported = true
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

	snapshot := *result
	return &snapshot, nil
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
