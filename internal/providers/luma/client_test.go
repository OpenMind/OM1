package luma

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// roundTripFunc lets us stub HTTP responses inline.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func stubClient(fn roundTripFunc) *http.Client {
	return &http.Client{Transport: fn}
}

func newTestClient(doer HTTPDoer, opts ...func(*Client)) *Client {
	c := NewClient("https://test.luma.com", "test-key", "evt-123", 5*time.Second, opts...)
	c.SetHTTPDoer(doer)
	return c
}

func jsonResp(code int, body any) *http.Response {
	b, _ := json.Marshal(body)
	return &http.Response{
		StatusCode: code,
		Body:       io.NopCloser(strings.NewReader(string(b))),
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}
}

func TestGetGuest_OK_Envelope(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		if !strings.Contains(req.URL.Path, GetGuestPath) {
			t.Fatalf("unexpected path: %s", req.URL.Path)
		}
		if req.Header.Get("x-luma-api-key") != "test-key" {
			t.Fatalf("missing api key header")
		}
		return jsonResp(200, guestEnvelope{
			Guest: &Guest{APIID: "gst-1", UserFirstName: "Alice", UserEmail: "alice@x.com"},
			Event: &struct {
				APIID string `json:"api_id"`
			}{APIID: "evt-123"},
		}), nil
	})

	c := newTestClient(doer)
	g, err := c.GetGuest(context.Background(), "g-abc")
	if err != nil {
		t.Fatal(err)
	}
	if g.APIID != "gst-1" {
		t.Fatalf("got APIID %q", g.APIID)
	}
	if g.UserFirstName != "Alice" {
		t.Fatalf("got first name %q", g.UserFirstName)
	}
	if g.EventAPIID != "evt-123" {
		t.Fatalf("got event api id %q", g.EventAPIID)
	}
}

func TestGetGuest_OK_Bare(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		return jsonResp(200, Guest{APIID: "gst-2", UserName: "Bob Smith"}), nil
	})

	c := newTestClient(doer)
	g, err := c.GetGuest(context.Background(), "g-abc")
	if err != nil {
		t.Fatal(err)
	}
	if g.APIID != "gst-2" {
		t.Fatalf("got APIID %q", g.APIID)
	}
}

func TestGetGuest_NotFound(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		return jsonResp(404, map[string]string{"error": "not found"}), nil
	})

	c := newTestClient(doer)
	_, err := c.GetGuest(context.Background(), "g-abc")
	if err == nil || err.Error() != ErrNotFound.Error() {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

func TestGetGuest_Unauthorized(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		return jsonResp(401, nil), nil
	})

	c := newTestClient(doer)
	_, err := c.GetGuest(context.Background(), "g-abc")
	if err == nil || err.Error() != ErrUnauthorized.Error() {
		t.Fatalf("expected ErrUnauthorized, got %v", err)
	}
}

func TestGetGuest_429_Retry(t *testing.T) {
	calls := 0
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return jsonResp(429, nil), nil
		}
		return jsonResp(200, guestEnvelope{
			Guest: &Guest{APIID: "gst-retry"},
		}), nil
	})

	c := newTestClient(doer)
	g, err := c.GetGuest(context.Background(), "g-abc")
	if err != nil {
		t.Fatal(err)
	}
	if g.APIID != "gst-retry" {
		t.Fatalf("got APIID %q", g.APIID)
	}
	if calls != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
}

func TestCheckIn_OK(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		if req.URL.String() != CheckInURL {
			t.Fatalf("unexpected url: %s", req.URL)
		}
		if !strings.Contains(req.Header.Get("Cookie"), "luma.auth-session-key=sess-abc") {
			t.Fatalf("missing session cookie")
		}
		body, _ := io.ReadAll(req.Body)
		var payload map[string]string
		_ = json.Unmarshal(body, &payload)
		if payload["rsvp_api_id"] != "gst-1" {
			t.Fatalf("wrong rsvp_api_id: %s", payload["rsvp_api_id"])
		}
		if payload["check_in_status"] != "checked-in" {
			t.Fatalf("wrong status: %s", payload["check_in_status"])
		}
		return jsonResp(200, map[string]string{"ok": "true"}), nil
	})

	c := newTestClient(doer, WithSessionKey("sess-abc"))
	err := c.CheckIn(context.Background(), &Guest{APIID: "gst-1"})
	if err != nil {
		t.Fatal(err)
	}
}

func TestCheckIn_NoSessionKey(t *testing.T) {
	c := NewClient("", "key", "evt-1", time.Second)
	err := c.CheckIn(context.Background(), &Guest{APIID: "gst-1"})
	if err == nil || !strings.Contains(err.Error(), "no session key") {
		t.Fatalf("expected no session key error, got %v", err)
	}
}

func TestCheckIn_NilGuest(t *testing.T) {
	c := NewClient("", "key", "evt-1", time.Second, WithSessionKey("s"))
	err := c.CheckIn(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "missing guest") {
		t.Fatalf("expected missing guest error, got %v", err)
	}
}

func TestCheckIn_ServerError(t *testing.T) {
	doer := stubClient(func(req *http.Request) (*http.Response, error) {
		return jsonResp(500, map[string]string{"error": "internal"}), nil
	})

	c := newTestClient(doer, WithSessionKey("sess"))
	err := c.CheckIn(context.Background(), &Guest{APIID: "gst-1"})
	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("expected 500 error, got %v", err)
	}
}

func TestFirstName(t *testing.T) {
	cases := []struct {
		name  string
		guest *Guest
		want  string
	}{
		{"nil guest", nil, "friend"},
		{"first name set", &Guest{UserFirstName: "Alice"}, "Alice"},
		{"only full name", &Guest{UserName: "Bob Smith"}, "Bob"},
		{"empty", &Guest{}, "friend"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := FirstName(tc.guest); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestFormatGreeting(t *testing.T) {
	g := &Guest{
		UserFirstName: "Alice",
		UserLastName:  "Smith",
		UserName:      "Alice Smith",
		UserEmail:     "alice@x.com",
	}
	tmpl := "Hi {first_name} {last_name}, welcome {name}! Contact: {email}"
	got := FormatGreeting(tmpl, g)
	want := "Hi Alice Smith, welcome Alice Smith! Contact: alice@x.com"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatGreeting_Nil(t *testing.T) {
	got := FormatGreeting("Hello {first_name}!", nil)
	if got != "Hello {first_name}!" {
		t.Fatalf("got %q", got)
	}
}

func TestNewClient_Defaults(t *testing.T) {
	c := NewClient("", "key", "evt-1", 0)
	if c.baseURL != DefaultBaseURL {
		t.Fatalf("got base url %q", c.baseURL)
	}
	if c.timeout != 5*time.Second {
		t.Fatalf("got timeout %v", c.timeout)
	}
}

func TestEventAPIID(t *testing.T) {
	c := NewClient("", "key", "evt-42", time.Second)
	if c.EventAPIID() != "evt-42" {
		t.Fatalf("got %q", c.EventAPIID())
	}
}
