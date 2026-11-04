package inputs

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

// newClassifierServer serves /classify with a fixed response and records what it received.
func newClassifierServer(t *testing.T, status int, resp map[string]any) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var got []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/classify", r.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		got = append(got, body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(srv.Close)
	return srv, &got
}

func writeLabel(t *testing.T, dir, label string) string {
	t.Helper()
	p := filepath.Join(dir, "label")
	require.NoError(t, os.WriteFile(p, []byte(label), 0o600))
	return p
}

func newFileSensor(t *testing.T, url, labelFile string, extra map[string]any) *HTTPClassifierSensor {
	t.Helper()
	cfg := map[string]any{"endpoint_url": url, "source": "file", "label_file": labelFile, "period_ms": 20, "timeout_ms": 500}
	for k, v := range extra {
		cfg[k] = v
	}
	s, err := NewHTTPClassifier(cfg)
	require.NoError(t, err)
	hs, ok := s.(*HTTPClassifierSensor)
	require.True(t, ok)
	hs.log = zap.NewNop()
	return hs
}

func TestNewHTTPClassifierConfig(t *testing.T) {
	s, err := NewHTTPClassifier(map[string]any{"source": "file", "label_file": "/tmp/x", "endpoint_url": "http://h:1/"})
	require.NoError(t, err)
	hs := s.(*HTTPClassifierSensor)
	require.Equal(t, "http://h:1", hs.cfg.EndpointURL, "trailing slash is trimmed")
	require.Equal(t, httpClassifierDefaultPeriod, hs.period, "period defaults when unset")
	require.Equal(t, httpClassifierDefaultTimeout, hs.client.Timeout)
	require.Nil(t, hs.paths, "file source needs no paths provider")
	require.True(t, hs.TriggersTick())

	_, err = NewHTTPClassifier(map[string]any{"source": "bogus"})
	require.Error(t, err)
	_, err = NewHTTPClassifier(map[string]any{"source": "file"})
	require.Error(t, err, "file source without label_file is rejected")
}

func TestHTTPClassifierLabelFromFile(t *testing.T) {
	dir := t.TempDir()
	p := writeLabel(t, dir, "a\n")
	s := newFileSensor(t, "http://127.0.0.1:1", p, nil)

	l, err := s.label()
	require.NoError(t, err)
	require.Equal(t, "A", l, "case-insensitive, first byte")

	writeLabel(t, dir, "")
	l, err = s.label()
	require.NoError(t, err)
	require.Equal(t, "B", l, "empty file means quiet")

	writeLabel(t, dir, "Z")
	_, err = s.label()
	require.Error(t, err)

	s.cfg.LabelFile = filepath.Join(dir, "missing")
	_, err = s.label()
	require.Error(t, err)
}

func TestHTTPClassifierPollPostsLabelAndParsesResponse(t *testing.T) {
	srv, got := newClassifierServer(t, http.StatusOK, map[string]any{"seq": 1, "label": "A", "confidence": 0.83, "latency_ms": 412.0, "extra_field": "ignored"})
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, srv.URL, p, nil)

	raw, err := s.Poll(context.Background())
	require.NoError(t, err)
	r, ok := raw.(*httpClassifierResponse)
	require.True(t, ok)
	require.Equal(t, "A", r.Label)
	require.InDelta(t, 0.83, r.Confidence, 1e-9)
	require.InDelta(t, 412.0, r.LatencyMs, 1e-9)

	require.Len(t, *got, 1)
	require.Equal(t, "A", (*got)[0]["symbol"])
	require.EqualValues(t, 1, (*got)[0]["seq"])
	require.NotZero(t, (*got)[0]["t_robot"])

	_, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.EqualValues(t, 2, (*got)[1]["seq"], "seq increments per request")
}

func TestHTTPClassifierPollErrors(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")

	refused, _ := newClassifierServer(t, http.StatusForbidden, map[string]any{"error": "not approved"})
	s := newFileSensor(t, refused.URL, p, nil)
	_, err := s.Poll(context.Background())
	require.Error(t, err)
	require.Contains(t, err.Error(), "not approved")

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("not json")) }))
	t.Cleanup(bad.Close)
	s = newFileSensor(t, bad.URL, p, nil)
	_, err = s.Poll(context.Background())
	require.Error(t, err)

	status := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]any{"label": "B"})
	}))
	t.Cleanup(status.Close)
	s = newFileSensor(t, status.URL, p, nil)
	_, err = s.Poll(context.Background())
	require.Error(t, err)
	require.Contains(t, err.Error(), "502")

	s = newFileSensor(t, "http://127.0.0.1:1", p, nil)
	_, err = s.Poll(context.Background())
	require.Error(t, err, "unreachable endpoint")

	s.cfg.LabelFile = filepath.Join(t.TempDir(), "missing")
	_, err = s.Poll(context.Background())
	require.Error(t, err, "label error is reported before any request")
}

func TestHTTPClassifierRawToText(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5})

	msg, err := s.RawToText(context.Background(), "not a response")
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 0.3})
	require.NoError(t, err)
	require.Nil(t, msg, "below min_confidence")

	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 0.9, LatencyMs: 300})
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Contains(t, msg.Message, "confirms change")
	require.Contains(t, msg.Message, "0.90")

	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "B", Confidence: 0.9})
	require.NoError(t, err)
	require.Nil(t, msg, "negative results are silent by default")

	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "abstain"})
	require.NoError(t, err)
	require.Nil(t, msg)

	s.cfg.ReportNegative = true
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "B", Confidence: 0.9})
	require.NoError(t, err)
	require.Contains(t, msg.Message, "no change")
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "abstain"})
	require.NoError(t, err)
	require.Contains(t, msg.Message, "no result")

	for i := 0; i < httpClassifierMaxMessages+5; i++ {
		_, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 1})
		require.NoError(t, err)
	}
	s.mu.Lock()
	require.Len(t, s.messages, httpClassifierMaxMessages, "buffer is bounded")
	s.mu.Unlock()
}

func TestHTTPClassifierOnlyOnChange(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5, "only_on_change": true})
	pos := &httpClassifierResponse{Label: "A", Confidence: 0.9}
	weak := &httpClassifierResponse{Label: "A", Confidence: 0.2}
	neg := &httpClassifierResponse{Label: "B", Confidence: 0.9}

	msg, err := s.RawToText(context.Background(), pos)
	require.NoError(t, err)
	require.NotNil(t, msg, "first positive is reported")
	msg, err = s.RawToText(context.Background(), pos)
	require.NoError(t, err)
	require.Nil(t, msg, "a persisting positive is not repeated")
	msg, err = s.RawToText(context.Background(), weak)
	require.NoError(t, err)
	require.Nil(t, msg, "a weak positive is never reported")
	msg, err = s.RawToText(context.Background(), pos)
	require.NoError(t, err)
	require.NotNil(t, msg, "a positive after a weak one starts a new episode")
	msg, err = s.RawToText(context.Background(), neg)
	require.NoError(t, err)
	require.Nil(t, msg, "negatives are silent unless report_negative")
	msg, err = s.RawToText(context.Background(), pos)
	require.NoError(t, err)
	require.NotNil(t, msg, "positive after negative is a new episode")

	s.cfg.ReportNegative = true
	msg, err = s.RawToText(context.Background(), neg)
	require.NoError(t, err)
	require.NotNil(t, msg, "with report_negative the transition to negative is reported once")
	msg, err = s.RawToText(context.Background(), neg)
	require.NoError(t, err)
	require.Nil(t, msg)
}

func TestHTTPClassifierFormattedLatestBuffer(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, nil)

	require.Equal(t, "", s.FormattedLatestBuffer(), "empty buffer formats to nothing")

	_, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 0.7, LatencyMs: 100})
	require.NoError(t, err)
	out := s.FormattedLatestBuffer()
	require.True(t, strings.HasPrefix(strings.TrimSpace(out), httpClassifierDescriptor))
	require.Contains(t, out, "confirms change")
	require.Equal(t, "", s.FormattedLatestBuffer(), "buffer is cleared after it is read")
}

func TestHTTPClassifierListenAndStop(t *testing.T) {
	srv, _ := newClassifierServer(t, http.StatusOK, map[string]any{"label": "A", "confidence": 1.0, "latency_ms": 1.0})
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, srv.URL, p, nil)

	ctx, cancel := context.WithCancel(context.Background())
	out, err := s.Listen(ctx)
	require.NoError(t, err)

	select {
	case raw := <-out:
		require.IsType(t, &httpClassifierResponse{}, raw)
	case <-time.After(2 * time.Second):
		t.Fatal("no reading within 2 s")
	}

	cancel()
	select {
	case _, open := <-out:
		for open {
			_, open = <-out
		}
	case <-time.After(2 * time.Second):
		t.Fatal("channel not closed after cancel")
	}
	s.mu.Lock()
	require.True(t, s.stopped)
	s.mu.Unlock()
	s.Stop() // idempotent
}

func TestHTTPClassifierListenKeepsPollingAfterErrors(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&n, 1) < 3 {
			w.WriteHeader(http.StatusForbidden)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": "not yet"})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"label": "A", "confidence": 1.0})
	}))
	t.Cleanup(srv.Close)
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, srv.URL, p, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := s.Listen(ctx)
	require.NoError(t, err)
	select {
	case raw := <-out:
		require.Equal(t, "A", raw.(*httpClassifierResponse).Label)
		require.GreaterOrEqual(t, atomic.LoadInt32(&n), int32(3), "refusals were retried, not fatal")
	case <-ctx.Done():
		t.Fatal("never recovered from refusals")
	}
}

func TestMatchSymbol(t *testing.T) {
	rules := []SymbolRule{{Match: []string{"waving", "smiling"}, Symbol: "A"}, {Match: []string{"turned away", "back to the camera"}, Symbol: "B"}}
	require.Equal(t, "A", matchSymbol("A man is Smiling at the camera", rules))
	require.Equal(t, "B", matchSymbol("the person has turned away from the desk", rules))
	require.Equal(t, "A", matchSymbol("waving while turned away", rules), "first matching rule wins")
	require.Equal(t, "", matchSymbol("a desk with two monitors", rules))
	require.Equal(t, "", matchSymbol("", rules))
}

func TestHTTPClassifierInputSourceConfig(t *testing.T) {
	_, err := NewHTTPClassifier(map[string]any{"source": "input"})
	require.Error(t, err, "input_key and symbol_rules are required")
	_, err = NewHTTPClassifier(map[string]any{"source": "input", "input_key": "VLMGemini", "symbol_rules": []any{map[string]any{"match": []any{"x"}, "symbol": "C"}}})
	require.Error(t, err, "symbols must be A or B")
	s, err := NewHTTPClassifier(map[string]any{"source": "input", "input_key": "VLMGemini", "symbol_rules": []any{map[string]any{"match": []any{"smiling"}, "symbol": "A"}}})
	require.NoError(t, err)
	require.Equal(t, "VLMGemini", s.(*HTTPClassifierSensor).cfg.InputKey)
}

func TestHTTPClassifierInputSourcePollsOnlyOnNewMatchingText(t *testing.T) {
	srv, got := newClassifierServer(t, http.StatusOK, map[string]any{"label": "A", "confidence": 1.0, "latency_ms": 1.0})
	cfg := map[string]any{"endpoint_url": srv.URL, "source": "input", "input_key": "TestCaption", "timeout_ms": 500,
		"symbol_rules": []any{map[string]any{"match": []any{"waving", "smiling"}, "symbol": "A"}, map[string]any{"match": []any{"turned away"}, "symbol": "B"}}}
	raw, err := NewHTTPClassifier(cfg)
	require.NoError(t, err)
	s := raw.(*HTTPClassifierSensor)
	s.log = zap.NewNop()
	t.Cleanup(func() { providers.IO().RemoveInput("TestCaption") })

	r, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, r, "no caption yet: no request")
	require.Empty(t, *got)

	providers.IO().AddInput("TestCaption", "a desk and a chair", time.Now())
	r, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, r, "caption matches no rule: no request")
	require.Empty(t, *got)

	providers.IO().AddInput("TestCaption", "a man smiling at the camera", time.Now())
	r, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.NotNil(t, r)
	require.Len(t, *got, 1)
	require.Equal(t, "A", (*got)[0]["symbol"])

	r, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.Nil(t, r, "the same caption is not classified twice")
	require.Len(t, *got, 1)

	providers.IO().AddInput("TestCaption", "the man has turned away", time.Now())
	r, err = s.Poll(context.Background())
	require.NoError(t, err)
	require.NotNil(t, r)
	require.Len(t, *got, 2)
	require.Equal(t, "B", (*got)[1]["symbol"])
}

func TestHTTPClassifierLabelsNameResults(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"report_negative": true, "labels": map[string]any{"A": "warmth", "B": "distance"}})
	msg, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 0.9})
	require.NoError(t, err)
	require.Contains(t, msg.Message, "reports: warmth (0.90")
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "B", Confidence: 0.8})
	require.NoError(t, err)
	require.Contains(t, msg.Message, "reports: distance (0.80")
}

func TestHTTPClassifierListenSkipsEmptyPolls(t *testing.T) {
	srv, got := newClassifierServer(t, http.StatusOK, map[string]any{"label": "A", "confidence": 1.0})
	cfg := map[string]any{"endpoint_url": srv.URL, "source": "input", "input_key": "TestCaption2", "period_ms": 20, "timeout_ms": 500,
		"symbol_rules": []any{map[string]any{"match": []any{"waving"}, "symbol": "A"}}}
	raw, err := NewHTTPClassifier(cfg)
	require.NoError(t, err)
	s := raw.(*HTTPClassifierSensor)
	s.log = zap.NewNop()
	t.Cleanup(func() { providers.IO().RemoveInput("TestCaption2") })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out, err := s.Listen(ctx)
	require.NoError(t, err)
	time.Sleep(100 * time.Millisecond)
	require.Empty(t, *got, "no caption: the loop makes no requests")
	providers.IO().AddInput("TestCaption2", "a person waving", time.Now())
	select {
	case r := <-out:
		require.Equal(t, "A", r.(*httpClassifierResponse).Label)
	case <-time.After(2 * time.Second):
		t.Fatal("no reading after a matching caption")
	}
}

func TestHTTPClassifierDisposition(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{
		"min_confidence": 0.5, "present_as": "disposition", "change_after": 2, "conversation_window_s": -1, "reaction_cooldown_s": -1,
		"labels": map[string]any{"A": "warm", "B": "muted"}, "events": map[string]any{"A": "someone turned towards you"},
	})
	providers.IO().SetDynamicVar("disposition", "")
	warm := &httpClassifierResponse{Label: "A", Confidence: 0.8}
	faint := &httpClassifierResponse{Label: "A", Confidence: 0.6}
	muted := &httpClassifierResponse{Label: "B", Confidence: 0.8}
	strong := &httpClassifierResponse{Label: "A", Confidence: 1.0}

	msg, err := s.RawToText(context.Background(), warm)
	require.NoError(t, err)
	require.Nil(t, msg, "one result does not change the disposition when change_after is 2")
	d, _ := providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "", d, "the variable is not set until the disposition is in force")
	msg, err = s.RawToText(context.Background(), warm)
	require.NoError(t, err)
	require.NotNil(t, msg, "the second consecutive result changes it")
	require.Equal(t, "someone turned towards you", msg.Message, "the configured event wording")
	d, _ = providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "warm", d)
	msg, err = s.RawToText(context.Background(), faint)
	require.NoError(t, err)
	require.Nil(t, msg, "the same disposition again is no event")
	d, _ = providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "warm, faintly", d, "but the strength follows the latest result")
	msg, err = s.RawToText(context.Background(), muted)
	require.NoError(t, err)
	require.Nil(t, msg, "one stray vote does not flip it")
	msg, err = s.RawToText(context.Background(), muted)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "your disposition changed to muted", msg.Message, "default wording when no event text is configured")
	d, _ = providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "muted", d)
	msg, err = s.RawToText(context.Background(), strong)
	require.NoError(t, err)
	require.NotNil(t, msg, "a strong (unanimous) vote flips the disposition on its own")
	d, _ = providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "warm, strongly", d)
	providers.IO().SetDynamicVar("disposition", "")
}

func TestHTTPClassifierDispositionInConversation(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5, "present_as": "disposition", "reaction_cooldown_s": -1,
		"labels": map[string]any{"A": "warm", "B": "muted"}, "events": map[string]any{"A": "someone turned towards you"},
		"conversation_events": map[string]any{"B": "the person you are talking with turned away or covered their face"}})
	providers.IO().SetDynamicVar("disposition", "")
	providers.IO().AddInput("Voice", "hey, how's it going?", time.Now())
	msg, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 1.0})
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "while you were talking, someone turned towards you", msg.Message, "default in-conversation wording")
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "B", Confidence: 1.0})
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "the person you are talking with turned away or covered their face", msg.Message, "configured in-conversation wording")
	providers.IO().AddInput("Voice", "hey, how's it going?", time.Now().Add(-2*time.Minute))
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 1.0})
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "someone turned towards you", msg.Message, "after the conversation has gone quiet, the arrival wording")
	providers.IO().SetDynamicVar("disposition", "")
}

func TestHTTPClassifierDispositionReactionCooldown(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5, "present_as": "disposition", "conversation_window_s": -1, "reaction_cooldown_s": 0.3})
	providers.IO().SetDynamicVar("disposition", "")
	providers.IO().MarkReaction()
	msg, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 1.0})
	require.NoError(t, err)
	require.Nil(t, msg, "no event right after the runtime reacted to something else")
	d, _ := providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "A, strongly", d, "the disposition still changed")
	time.Sleep(350 * time.Millisecond)
	msg, err = s.RawToText(context.Background(), &httpClassifierResponse{Label: "B", Confidence: 1.0})
	require.NoError(t, err)
	require.NotNil(t, msg, "after the cooldown, changes are events again")
	providers.IO().SetDynamicVar("disposition", "")
}

func TestHTTPClassifierDispositionSilent(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5, "present_as": "disposition", "silent": true, "labels": map[string]any{"A": "warm"}})
	providers.IO().SetDynamicVar("disposition", "")
	msg, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: "A", Confidence: 1.0})
	require.NoError(t, err)
	require.Nil(t, msg, "silent: no event, ever")
	d, _ := providers.IO().GetDynamicVar("disposition")
	require.Equal(t, "warm, strongly", d, "only the disposition")
	providers.IO().SetDynamicVar("disposition", "")
}

func TestHTTPClassifierDispositionSmooth(t *testing.T) {
	p := writeLabel(t, t.TempDir(), "A")
	s := newFileSensor(t, "http://127.0.0.1:1", p, map[string]any{"min_confidence": 0.5, "present_as": "disposition", "silent": true, "smooth": 5, "labels": map[string]any{"A": "warm", "B": "muted"}})
	providers.IO().SetDynamicVar("disposition", "")
	vote := func(label string) {
		_, err := s.RawToText(context.Background(), &httpClassifierResponse{Label: label, Confidence: 1.0})
		require.NoError(t, err)
	}
	get := func() string { d, _ := providers.IO().GetDynamicVar("disposition"); return d }
	vote("A")
	vote("A")
	vote("A")
	vote("A")
	vote("A")
	require.Equal(t, "warm, strongly", get(), "five of five")
	vote("B")
	require.Equal(t, "warm", get(), "four of five: still warm, no longer strongly")
	vote("B")
	require.Equal(t, "warm, faintly", get(), "three of five")
	vote("B")
	vote("B")
	require.Equal(t, "muted", get(), "four of five the other way")
	vote("B")
	require.Equal(t, "muted, strongly", get())
	providers.IO().SetDynamicVar("disposition", "")
}
