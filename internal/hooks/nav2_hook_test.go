package hooks

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestNav2BaseURL(t *testing.T) {
	require.Equal(t, "http://example.com", nav2BaseURL(map[string]any{"base_url": "http://example.com"}),
		"explicit base_url wins")
	require.Equal(t, nav2CloudBaseURL, nav2BaseURL(map[string]any{"use_sim": true}),
		"use_sim falls back to the cloud orchestrator")
	require.Equal(t, nav2LocalBaseURL, nav2BaseURL(map[string]any{"use_sim": false}),
		"defaults to localhost when not simulating")
	require.Equal(t, nav2LocalBaseURL, nav2BaseURL(map[string]any{}),
		"defaults to localhost when nothing is set")
}

func TestStartNav2Hook(t *testing.T) {
	var gotPath, gotKey, gotMapName string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-api-key")
		var body struct {
			MapName string `json:"map_name"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotMapName = body.MapName
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.startNav2Hook(context.Background(), map[string]any{
		"base_url": srv.URL,
		"api_key":  "secret",
		"map_name": "office",
	}, nil)

	require.NoError(t, err)
	require.Equal(t, "/start/nav2", gotPath)
	require.Equal(t, "secret", gotKey)
	require.Equal(t, "office", gotMapName)
}

func TestStartNav2HookDefaultMapName(t *testing.T) {
	var gotMapName string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			MapName string `json:"map_name"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotMapName = body.MapName
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	require.NoError(t, runner.startNav2Hook(context.Background(), map[string]any{"base_url": srv.URL}, nil))
	require.Equal(t, nav2DefaultMap, gotMapName, "map_name defaults to \"map\"")
}

func TestStartNav2HookError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.startNav2Hook(context.Background(), map[string]any{"base_url": srv.URL}, nil)
	require.ErrorContains(t, err, "boom")
}

func TestStopNav2Hook(t *testing.T) {
	var gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-api-key")
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.stopNav2Hook(context.Background(), map[string]any{
		"base_url": srv.URL,
		"api_key":  "secret",
	}, nil)

	require.NoError(t, err)
	require.Equal(t, "/stop/nav2", gotPath)
	require.Equal(t, "secret", gotKey)
}

func TestStopNav2HookError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"message":"nope"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.stopNav2Hook(context.Background(), map[string]any{"base_url": srv.URL}, nil)
	require.ErrorContains(t, err, "nope")
}

func TestNav2HooksRegistered(t *testing.T) {
	_, ok := lookupHook("nav2_hook", "start_nav2_hook")
	require.True(t, ok)
	_, ok = lookupHook("nav2_hook", "stop_nav2_hook")
	require.True(t, ok)
}
