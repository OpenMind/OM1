package luma_checkin_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/openmind/om1/plugins/actions/luma_checkin"
)

type fakeGuestClient struct {
	guest          *luma_checkin.Guest
	getErr         error
	updateErr      error
	getCalls       int32
	updateCalls    int32
	lastStatusArgs struct {
		eventAPIID, guestAPIID, status string
	}
	mu sync.Mutex
}

func (f *fakeGuestClient) GetGuest(_ context.Context, _ string) (*luma_checkin.Guest, error) {
	atomic.AddInt32(&f.getCalls, 1)
	return f.guest, f.getErr
}

func (f *fakeGuestClient) UpdateGuestStatus(_ context.Context, eventAPIID, guestAPIID, status string) error {
	atomic.AddInt32(&f.updateCalls, 1)
	f.mu.Lock()
	f.lastStatusArgs.eventAPIID = eventAPIID
	f.lastStatusArgs.guestAPIID = guestAPIID
	f.lastStatusArgs.status = status
	f.mu.Unlock()
	return f.updateErr
}

type fakeTTS struct {
	mu    sync.Mutex
	texts []string
}

func (f *fakeTTS) AddText(t string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.texts = append(f.texts, t)
}

func (f *fakeTTS) lastText() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.texts) == 0 {
		return ""
	}
	return f.texts[len(f.texts)-1]
}

func newTestConnector(client luma_checkin.GuestLookup, cfg luma_checkin.Config) (*luma_checkin.TestConnector, *fakeTTS) {
	tts := &fakeTTS{}
	c := luma_checkin.NewTestConnector(client, cfg, tts)
	return c, tts
}

func defaultCfg() luma_checkin.Config {
	return luma_checkin.Config{
		APIKey:           "k",
		EventAPIID:       "evt-1",
		GreetingTemplate: luma_checkin.DefaultGreetingTemplate,
		StatusValue:      luma_checkin.DefaultStatus,
	}
}

func TestConnectHappyPath(t *testing.T) {
	client := &fakeGuestClient{guest: &luma_checkin.Guest{APIID: "g-1", FirstName: "Ada", EventAPIID: "evt-1"}}
	c, tts := newTestConnector(client, defaultCfg())

	out, err := c.Connect(context.Background(), map[string]any{"pk": "g-1"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if got, want := out.(string), "checked_in: Ada"; got != want {
		t.Errorf("output: got %q want %q", got, want)
	}
	if atomic.LoadInt32(&client.updateCalls) != 1 {
		t.Errorf("expected 1 update call, got %d", client.updateCalls)
	}
	if client.lastStatusArgs.guestAPIID != "g-1" || client.lastStatusArgs.status != luma_checkin.DefaultStatus {
		t.Errorf("update args: %+v", client.lastStatusArgs)
	}
	if !strings.Contains(tts.lastText(), "Ada") {
		t.Errorf("tts text missing first name: %q", tts.lastText())
	}
}

func TestConnectEventMismatch(t *testing.T) {
	client := &fakeGuestClient{guest: &luma_checkin.Guest{APIID: "g-1", FirstName: "Ada", EventAPIID: "evt-other"}}
	c, tts := newTestConnector(client, defaultCfg())

	out, err := c.Connect(context.Background(), map[string]any{"pk": "g-1"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if !strings.HasPrefix(out.(string), "checkin_failed:") {
		t.Errorf("output: %q", out)
	}
	if atomic.LoadInt32(&client.updateCalls) != 0 {
		t.Errorf("update should not be called on mismatch")
	}
	if tts.lastText() != "" {
		t.Errorf("tts should not fire on mismatch")
	}
}

func TestConnectStatusUpdateFailureNonBlocking(t *testing.T) {
	client := &fakeGuestClient{
		guest:     &luma_checkin.Guest{APIID: "g-1", FirstName: "Ada", EventAPIID: "evt-1"},
		updateErr: errors.New("boom"),
	}
	c, tts := newTestConnector(client, defaultCfg())

	out, err := c.Connect(context.Background(), map[string]any{"pk": "g-1"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	s := out.(string)
	if !strings.HasPrefix(s, "checked_in: Ada") || !strings.Contains(s, "status update failed") {
		t.Errorf("output: %q", s)
	}
	if !strings.Contains(tts.lastText(), "Ada") {
		t.Errorf("tts should still fire when status update fails: %q", tts.lastText())
	}
}

func TestConnectGuestNotFound(t *testing.T) {
	client := &fakeGuestClient{getErr: luma_checkin.ErrLumaNotFound}
	c, _ := newTestConnector(client, defaultCfg())

	out, err := c.Connect(context.Background(), map[string]any{"pk": "g-x"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if !strings.HasPrefix(out.(string), "checkin_failed:") {
		t.Errorf("output: %q", out)
	}
}

func TestConnectAuthFailure(t *testing.T) {
	client := &fakeGuestClient{getErr: luma_checkin.ErrLumaUnauthorized}
	c, _ := newTestConnector(client, defaultCfg())

	_, err := c.Connect(context.Background(), map[string]any{"pk": "g-1"})
	if err == nil {
		t.Fatalf("expected error on auth failure")
	}
}

func TestConnectEmptyPk(t *testing.T) {
	client := &fakeGuestClient{}
	c, _ := newTestConnector(client, defaultCfg())

	_, err := c.Connect(context.Background(), map[string]any{"pk": ""})
	if err == nil {
		t.Fatalf("expected error on empty pk")
	}
	if atomic.LoadInt32(&client.getCalls) != 0 {
		t.Errorf("should not call GetGuest with empty pk")
	}
}

func TestFirstNameFallbacks(t *testing.T) {
	cases := []struct {
		guest *luma_checkin.Guest
		want  string
	}{
		{&luma_checkin.Guest{FirstName: "Ada"}, "Ada"},
		{&luma_checkin.Guest{Name: "Ada Lovelace"}, "Ada"},
		{&luma_checkin.Guest{}, "friend"},
	}
	for _, tc := range cases {
		if got := luma_checkin.FirstNameFor(tc.guest); got != tc.want {
			t.Errorf("FirstNameFor(%+v): got %q want %q", tc.guest, got, tc.want)
		}
	}
}

func TestFormatGreetingTokens(t *testing.T) {
	g := &luma_checkin.Guest{FirstName: "Ada", LastName: "Lovelace", Name: "Ada Lovelace", Email: "ada@example.com"}
	got := luma_checkin.FormatGreeting("Hi {first_name} {last_name} <{email}>", g)
	want := "Hi Ada Lovelace <ada@example.com>"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}
