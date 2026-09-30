package vlm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestDescribeMergesExtraBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		assert.Equal(t, map[string]any{"enable_thinking": false}, body["chat_template_kwargs"])
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"two people"}}]}`))
	}))
	defer srv.Close()

	d := NewDescriber(Describer{
		BaseURL:   srv.URL,
		ExtraBody: map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}},
		Log:       zap.NewNop(),
	})
	text, err := d.Describe(context.Background(), "")
	require.NoError(t, err)
	assert.Equal(t, "two people", text)
}

func TestDescribeTimeout(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		<-release
	}))
	defer srv.Close()
	defer close(release)

	d := NewDescriber(Describer{BaseURL: srv.URL, Timeout: 50 * time.Millisecond, Log: zap.NewNop()})
	_, err := d.Describe(context.Background(), "")
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestDescribeWarmsUpInBackground(t *testing.T) {
	prev := warmupRetryDelay
	warmupRetryDelay = time.Millisecond
	defer func() { warmupRetryDelay = prev }()

	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer srv.Close()

	d := NewDescriber(Describer{BaseURL: srv.URL, Warmup: true, Log: zap.NewNop()})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	text, err := d.Describe(ctx, "")
	require.NoError(t, err)
	assert.Empty(t, text)

	require.Eventually(t, func() bool {
		text, err := d.Describe(ctx, "")
		return err == nil && text == "ok"
	}, 2*time.Second, 5*time.Millisecond)
}

func TestWarmupRestartsAfterCancel(t *testing.T) {
	d := NewDescriber(Describer{BaseURL: "http://127.0.0.1:1", Warmup: true, Log: zap.NewNop()})
	ctx, cancel := context.WithCancel(context.Background())

	_, _ = d.Describe(ctx, "")
	cancel()

	require.Eventually(t, func() bool { return !d.warm.started.Load() }, 5*time.Second, 5*time.Millisecond)
	assert.False(t, d.warm.ready.Load())
}
