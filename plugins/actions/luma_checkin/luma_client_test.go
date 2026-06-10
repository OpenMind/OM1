package luma_checkin_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openmind/om1/plugins/actions/luma_checkin"
)

func newTestClient(t *testing.T, handler http.HandlerFunc) (*luma_checkin.LumaClient, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	c := luma_checkin.NewLumaClient(srv.URL, "test-key", "evt-test", 2*time.Second)
	return c, srv
}

func TestGetGuestEnvelopeOK(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != luma_checkin.GetGuestPath {
			t.Errorf("path: got %q want %q", r.URL.Path, luma_checkin.GetGuestPath)
		}
		if r.URL.Query().Get("id") != "g-123" {
			t.Errorf("id query: got %q want g-123", r.URL.Query().Get("id"))
		}
		if r.URL.Query().Get("event_id") != "evt-test" {
			t.Errorf("event_id query: got %q want evt-test", r.URL.Query().Get("event_id"))
		}
		if r.Header.Get("x-luma-api-key") != "test-key" {
			t.Errorf("missing api key header: %q", r.Header.Get("x-luma-api-key"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"guest":{"api_id":"g-123","first_name":"Ada","name":"Ada Lovelace","email":"ada@example.com"},"event":{"api_id":"evt-abc"}}`)
	})
	defer srv.Close()

	g, err := c.GetGuest(context.Background(), "g-123")
	if err != nil {
		t.Fatalf("GetGuest: %v", err)
	}
	if g.APIID != "g-123" || g.FirstName != "Ada" || g.EventAPIID != "evt-abc" {
		t.Errorf("unexpected guest: %+v", g)
	}
}

func TestGetGuestBareOK(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"api_id":"g-9","first_name":"Bo","event_api_id":"evt-1"}`)
	})
	defer srv.Close()

	g, err := c.GetGuest(context.Background(), "g-9")
	if err != nil {
		t.Fatalf("GetGuest: %v", err)
	}
	if g.FirstName != "Bo" || g.EventAPIID != "evt-1" {
		t.Errorf("unexpected guest: %+v", g)
	}
}

func TestGetGuestNotFound(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	})
	defer srv.Close()

	_, err := c.GetGuest(context.Background(), "g-x")
	if !errors.Is(err, luma_checkin.ErrLumaNotFound) {
		t.Fatalf("want ErrLumaNotFound, got %v", err)
	}
}

func TestGetGuestUnauthorized(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusForbidden} {
		code := code
		c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(code)
		})
		_, err := c.GetGuest(context.Background(), "g-x")
		srv.Close()
		if !errors.Is(err, luma_checkin.ErrLumaUnauthorized) {
			t.Errorf("status %d: want ErrLumaUnauthorized, got %v", code, err)
		}
	}
}

func TestGetGuestRetriesOn429(t *testing.T) {
	var calls int
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = io.WriteString(w, `{"api_id":"g-1","first_name":"A","event_api_id":"e"}`)
	})
	defer srv.Close()

	g, err := c.GetGuest(context.Background(), "g-1")
	if err != nil {
		t.Fatalf("GetGuest: %v", err)
	}
	if calls != 2 {
		t.Errorf("expected 2 attempts, got %d", calls)
	}
	if g.FirstName != "A" {
		t.Errorf("unexpected guest: %+v", g)
	}
}

func TestUpdateGuestStatusOK(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method: got %s want POST", r.Method)
		}
		if r.URL.Path != luma_checkin.UpdateGuestStatusPath {
			t.Errorf("path: got %q want %q", r.URL.Path, luma_checkin.UpdateGuestStatusPath)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type: %q", r.Header.Get("Content-Type"))
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if body["event_id"] != "evt-1" || body["id"] != "g-1" || body["status"] != "checked_in" {
			t.Errorf("body: %+v", body)
		}
		w.WriteHeader(http.StatusOK)
	})
	defer srv.Close()

	if err := c.UpdateGuestStatus(context.Background(), "evt-1", "g-1", "checked_in"); err != nil {
		t.Fatalf("UpdateGuestStatus: %v", err)
	}
}

func TestUpdateGuestStatusErrorBubbles(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":"bad shape"}`)
	})
	defer srv.Close()

	err := c.UpdateGuestStatus(context.Background(), "evt-1", "g-1", "checked_in")
	if err == nil || !strings.Contains(err.Error(), "400") {
		t.Fatalf("want 400 error, got %v", err)
	}
}
