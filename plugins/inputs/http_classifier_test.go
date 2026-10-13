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
