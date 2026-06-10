package luma_checkin

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	defaultLumaBaseURL    = "https://public-api.luma.com"
	getGuestPath          = "/v1/event/get-guest"
	updateGuestStatusPath = "/v1/event/update-guest-status"
)

var (
	errLumaNotFound     = errors.New("luma: guest not found")
	errLumaUnauthorized = errors.New("luma: unauthorized")
)

// Guest is a flattened view of the guest record returned by Luma. Field names
// follow the API's snake_case JSON. Unknown fields are ignored.
type Guest struct {
	APIID          string `json:"api_id"`
	Name           string `json:"name"`
	FirstName      string `json:"first_name"`
	LastName       string `json:"last_name"`
	Email          string `json:"email"`
	EventAPIID     string `json:"event_api_id"`
	CheckedInAt    string `json:"checked_in_at"`
	ApprovalStatus string `json:"approval_status"`
}

// guestEnvelope handles Luma's nested response shape: {"guest": {...}, "event": {...}}.
// Some endpoints return the guest object directly; we try the envelope first.
type guestEnvelope struct {
	Guest *Guest `json:"guest"`
	Event *struct {
		APIID string `json:"api_id"`
	} `json:"event"`
}

// httpDoer is satisfied by *http.Client; abstracted for tests.
type httpDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

type lumaClient struct {
	baseURL    string
	apiKey     string
	eventAPIID string
	http       httpDoer
	timeout    time.Duration
}

func newLumaClient(baseURL, apiKey, eventAPIID string, timeout time.Duration) *lumaClient {
	if baseURL == "" {
		baseURL = defaultLumaBaseURL
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	return &lumaClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		eventAPIID: eventAPIID,
		http:       httpclient.Default(),
		timeout:    timeout,
	}
}

// GetGuest fetches the guest record for the given key (g-<id> or ticket key).
// Luma's get-guest requires both event_id (the configured event) and id (the
// guest/ticket key from the QR).
func (c *lumaClient) GetGuest(ctx context.Context, pk string) (*Guest, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	q := url.Values{
		"event_id": []string{c.eventAPIID},
		"id":       []string{pk},
	}
	endpoint := c.baseURL + getGuestPath + "?" + q.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	switch resp.StatusCode {
	case http.StatusOK:
		// fallthrough to decode
	case http.StatusNotFound:
		return nil, errLumaNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, errLumaUnauthorized
	default:
		return nil, fmt.Errorf("luma get-guest %d: %s", resp.StatusCode, string(body))
	}

	var env guestEnvelope
	if err := json.Unmarshal(body, &env); err == nil && env.Guest != nil {
		g := env.Guest
		if g.EventAPIID == "" && env.Event != nil {
			g.EventAPIID = env.Event.APIID
		}
		return g, nil
	}
	var bare Guest
	if err := json.Unmarshal(body, &bare); err != nil {
		return nil, fmt.Errorf("luma get-guest decode: %w", err)
	}
	return &bare, nil
}

// UpdateGuestStatus flips the guest's approval/check-in status. Best-effort: the
// exact body shape Luma expects is under-documented, so the action treats
// failures here as non-fatal.
func (c *lumaClient) UpdateGuestStatus(ctx context.Context, eventAPIID, guestAPIID, status string) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	body, _ := json.Marshal(map[string]any{
		"event_id": eventAPIID,
		"id":       guestAPIID,
		"status":   status,
	})
	endpoint := c.baseURL + updateGuestStatusPath
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	resp, err := c.do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	rb, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("luma update-guest-status %d: %s", resp.StatusCode, string(rb))
}

func (c *lumaClient) setAuth(req *http.Request) {
	req.Header.Set("x-luma-api-key", c.apiKey)
	req.Header.Set("Accept", "application/json")
}

// do issues req with one retry on 429 after a short backoff.
func (c *lumaClient) do(req *http.Request) (*http.Response, error) {
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		return resp, nil
	}

	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	t := time.NewTimer(500 * time.Millisecond)
	defer t.Stop()
	select {
	case <-req.Context().Done():
		return nil, req.Context().Err()
	case <-t.C:
	}

	if req.GetBody != nil {
		body, berr := req.GetBody()
		if berr != nil {
			return nil, berr
		}
		req.Body = body
	}
	return c.http.Do(req)
}
