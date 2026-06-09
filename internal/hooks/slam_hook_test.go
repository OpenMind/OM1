package hooks

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestSlamBaseURL(t *testing.T) {
	require.Equal(t, "http://example.com", slamBaseURL(map[string]any{"base_url": "http://example.com"}),
		"explicit base_url wins")
	require.Equal(t, slamCloudBaseURL, slamBaseURL(map[string]any{"use_sim": true}),
		"use_sim falls back to the cloud orchestrator")
	require.Equal(t, slamLocalBaseURL, slamBaseURL(map[string]any{"use_sim": false}),
		"defaults to localhost when not simulating")
	require.Equal(t, slamLocalBaseURL, slamBaseURL(map[string]any{}),
		"defaults to localhost when nothing is set")
}

func TestStartSlamHook(t *testing.T) {
	var gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotKey = r.Header.Get("x-api-key")
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.startSlamHook(context.Background(), map[string]any{
		"base_url": srv.URL,
		"api_key":  "secret",
	}, nil)

	require.NoError(t, err)
	require.Equal(t, "/start/slam", gotPath)
	require.Equal(t, "secret", gotKey)
}

func TestStartSlamHookError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"message":"boom"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.startSlamHook(context.Background(), map[string]any{"base_url": srv.URL}, nil)
	require.ErrorContains(t, err, "boom")
}

func TestStopSlamHook(t *testing.T) {
	var mu sync.Mutex
	paths := []string{}
	var savedMapName string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		if r.URL.Path == "/maps/save" {
			var body struct {
				MapName string `json:"map_name"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			savedMapName = body.MapName
		}
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.stopSlamHook(context.Background(), map[string]any{
		"base_url": srv.URL,
		"map_name": "office",
	}, nil)

	require.NoError(t, err)
	require.Equal(t, []string{"/maps/save", "/stop/slam"}, paths, "save runs before stop")
	require.Equal(t, "office", savedMapName)
}

func TestStopSlamHookDefaultMapName(t *testing.T) {
	var savedMapName string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/maps/save" {
			var body struct {
				MapName string `json:"map_name"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			savedMapName = body.MapName
		}
		_, _ = io.WriteString(w, `{"message":"ok"}`)
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	require.NoError(t, runner.stopSlamHook(context.Background(), map[string]any{"base_url": srv.URL}, nil))
	require.Equal(t, slamDefaultMap, savedMapName, "map_name defaults to \"map\"")
}

func TestStopSlamHookSaveFailureSkipsStop(t *testing.T) {
	var mu sync.Mutex
	paths := []string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		if r.URL.Path == "/maps/save" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"message":"save failed"}`)
		}
	}))
	defer srv.Close()

	runner := NewHooks(nil, zap.NewNop())
	err := runner.stopSlamHook(context.Background(), map[string]any{"base_url": srv.URL}, nil)
	require.ErrorContains(t, err, "save failed")
	require.Equal(t, []string{"/maps/save"}, paths, "stop is not called when save fails")
}

func TestSlamHooksRegistered(t *testing.T) {
	_, ok := lookupHook("slam_hook", "start_slam_hook")
	require.True(t, ok)
	_, ok = lookupHook("slam_hook", "stop_slam_hook")
	require.True(t, ok)
}
