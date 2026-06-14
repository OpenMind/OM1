// Package luma provides a thin client for Luma's public API. Currently it only
// exposes /v1/event/get-guest, which is enough to look up a guest by the pk
// embedded in a check-in QR code and surface their name to the rest of OM1.
package luma

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"bytes"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	DefaultBaseURL     = "https://public-api.luma.com"
	GetGuestPath       = "/v1/event/get-guest"
	CheckInURL         = "https://api.luma.com/event/admin/update-check-in"
)

var (
	ErrNotFound     = errors.New("luma: guest not found")
	ErrUnauthorized = errors.New("luma: unauthorized")
)

// Guest is a flattened view of the guest record returned by Luma's
// /v1/event/get-guest. Identity fields use the user_-prefixed names from the
// public API schema. Unknown fields are ignored.
type Guest struct {
	APIID          string `json:"api_id"`
	UserAPIID      string `json:"user_api_id"`
	UserName       string `json:"user_name"`
	UserFirstName  string `json:"user_first_name"`
	UserLastName   string `json:"user_last_name"`
	UserEmail      string `json:"user_email"`
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

// HTTPDoer is satisfied by *http.Client; abstracted for tests.
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

type Client struct {
	baseURL    string
	apiKey     string
	eventAPIID string
	sessionKey string
	http       HTTPDoer
	timeout    time.Duration
}

func NewClient(baseURL, apiKey, eventAPIID string, timeout time.Duration, opts ...func(*Client)) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		eventAPIID: eventAPIID,
		http:       httpclient.Default(),
		timeout:    timeout,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

func WithSessionKey(key string) func(*Client) {
	return func(c *Client) { c.sessionKey = key }
}

// SetHTTPDoer overrides the underlying http client. Intended for tests.
func (c *Client) SetHTTPDoer(d HTTPDoer) { c.http = d }

// EventAPIID returns the configured event id.
func (c *Client) EventAPIID() string { return c.eventAPIID }

// GetGuest fetches the guest record for the given key (g-<id> or ticket key).
// Luma's get-guest requires both event_id (the configured event) and id (the
// guest/ticket key from the QR).
func (c *Client) GetGuest(ctx context.Context, pk string) (*Guest, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	q := url.Values{
		"event_id": []string{c.eventAPIID},
		"id":       []string{pk},
	}
	endpoint := c.baseURL + GetGuestPath + "?" + q.Encode()
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

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("luma get-guest read body: %w", err)
	}
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return nil, ErrNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, ErrUnauthorized
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

func (c *Client) setAuth(req *http.Request) {
	req.Header.Set("x-luma-api-key", c.apiKey)
	req.Header.Set("Accept", "application/json")
}

// do issues req with one retry on 429 after a short backoff.
func (c *Client) do(req *http.Request) (*http.Response, error) {
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

// CheckIn marks a guest as checked-in via Luma's admin endpoint.
// Requires a session key (from a logged-in browser session).
func (c *Client) CheckIn(ctx context.Context, guest *Guest) error {
	if c.sessionKey == "" {
		return fmt.Errorf("luma check-in: no session key configured")
	}
	if guest == nil || guest.APIID == "" {
		return fmt.Errorf("luma check-in: missing guest api_id")
	}

	payload := map[string]string{
		"event_api_id":    c.eventAPIID,
		"rsvp_api_id":     guest.APIID,
		"check_in_method": "guest-list",
		"check_in_status": "checked-in",
		"type":            "guest",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("luma check-in marshal: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, CheckInURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("luma check-in request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Cookie", "luma.auth-session-key="+c.sessionKey)

	resp, err := c.do(req)
	if err != nil {
		return fmt.Errorf("luma check-in: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("luma check-in %d: %s", resp.StatusCode, string(respBody))
	}
	return nil
}

// FirstName picks the best available first-name field from a Guest record.
// Falls back to "friend" when nothing usable is set.
func FirstName(g *Guest) string {
	if g == nil {
		return "friend"
	}
	if g.UserFirstName != "" {
		return g.UserFirstName
	}
	if g.UserName != "" {
		if parts := strings.Fields(g.UserName); len(parts) > 0 {
			return parts[0]
		}
	}
	return "friend"
}

// FormatGreeting interpolates {first_name}, {last_name}, {name}, {email} into
// template using the guest's fields.
func FormatGreeting(template string, g *Guest) string {
	if g == nil {
		return template
	}
	name := g.UserName
	if name == "" {
		name = FirstName(g)
	}
	r := strings.NewReplacer(
		"{first_name}", FirstName(g),
		"{last_name}", g.UserLastName,
		"{name}", name,
		"{email}", g.UserEmail,
	)
	return r.Replace(template)
}
