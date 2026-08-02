package home_assistant

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func newMockHAStatesServer(t *testing.T, states map[string]map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// path: /api/states/{entity_id}
		entityID := r.URL.Path[len("/api/states/"):]
		state, ok := states[entityID]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(state)
	}))
}

func buildStateSensor(t *testing.T, baseURL string, entityIDs string, pollInterval float64) *StateSensor {
	t.Helper()
	sensor, err := NewHomeAssistantInput(map[string]any{
		"base_url":      baseURL,
		"token":         "test-token",
		"entity_ids":    entityIDs,
		"poll_interval": pollInterval,
	})
	require.NoError(t, err)
	s, ok := sensor.(*StateSensor)
	require.True(t, ok, "expected *StateSensor")
	return s
}

func TestHomeAssistantInputConfig(t *testing.T) {
	t.Run("default values", func(t *testing.T) {
		sensor, err := NewHomeAssistantInput(map[string]any{
			"base_url": "http://example.com",
			"token":    "tok",
		})
		require.NoError(t, err)
		s := sensor.(*StateSensor)
		require.Equal(t, time.Duration(defaultInterval*float64(time.Second)), s.pollInterval)
	})

	t.Run("custom values", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a, switch.b ,climate.c", 5.0)
		require.Equal(t, []string{"light.a", "switch.b", "climate.c"}, s.entityIDs)
		require.Equal(t, 5*time.Second, s.pollInterval)
	})

	t.Run("warns but does not fail when base_url missing", func(t *testing.T) {
		sensor, err := NewHomeAssistantInput(map[string]any{"token": "tok", "entity_ids": "light.a"})
		require.NoError(t, err)
		require.NotNil(t, sensor)
	})

	t.Run("warns but does not fail when token missing", func(t *testing.T) {
		sensor, err := NewHomeAssistantInput(map[string]any{"base_url": "http://example.com", "entity_ids": "light.a"})
		require.NoError(t, err)
		require.NotNil(t, sensor)
	})

	t.Run("warns but does not fail when entity_ids missing", func(t *testing.T) {
		sensor, err := NewHomeAssistantInput(map[string]any{"base_url": "http://example.com", "token": "tok"})
		require.NoError(t, err)
		require.NotNil(t, sensor)
	})
}

func TestFetchState(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		server := newMockHAStatesServer(t, map[string]map[string]any{
			"light.bed_light": {
				"entity_id": "light.bed_light",
				"state":     "on",
				"attributes": map[string]any{
					"friendly_name": "Bed Light",
					"brightness":    128.0,
				},
			},
		})
		defer server.Close()

		s := buildStateSensor(t, server.URL, "light.bed_light", 0)
		state, err := s.fetchState(context.Background(), "light.bed_light")
		require.NoError(t, err)
		require.Equal(t, "on", state["state"])
	})

	t.Run("error status", func(t *testing.T) {
		server := newMockHAStatesServer(t, map[string]map[string]any{})
		defer server.Close()

		s := buildStateSensor(t, server.URL, "light.missing", 0)
		state, err := s.fetchState(context.Background(), "light.missing")
		require.Error(t, err)
		require.Nil(t, state)
	})

	t.Run("no base_url returns nil, nil", func(t *testing.T) {
		s := buildStateSensor(t, "", "light.a", 0)
		s.token = "tok"
		state, err := s.fetchState(context.Background(), "light.a")
		require.NoError(t, err)
		require.Nil(t, state)
	})

	t.Run("no token returns nil, nil", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		s.token = ""
		state, err := s.fetchState(context.Background(), "light.a")
		require.NoError(t, err)
		require.Nil(t, state)
	})

	t.Run("unreachable server", func(t *testing.T) {
		s := buildStateSensor(t, "http://127.0.0.1:1", "light.a", 0)
		state, err := s.fetchState(context.Background(), "light.a")
		require.Error(t, err)
		require.Nil(t, state)
	})
}

func TestPoll(t *testing.T) {
	t.Run("returns nil before interval elapses", func(t *testing.T) {
		server := newMockHAStatesServer(t, map[string]map[string]any{
			"light.a": {"entity_id": "light.a", "state": "on"},
		})
		defer server.Close()

		s := buildStateSensor(t, server.URL, "light.a", 30)
		// First poll triggers fetch and sets lastPollTime = now.
		_, err := s.Poll(context.Background())
		require.NoError(t, err)

		// Second poll immediately after should be throttled.
		raw, err := s.Poll(context.Background())
		require.NoError(t, err)
		require.Nil(t, raw)
	})

	t.Run("fetches after interval elapses", func(t *testing.T) {
		server := newMockHAStatesServer(t, map[string]map[string]any{
			"light.a": {"entity_id": "light.a", "state": "on"},
		})
		defer server.Close()

		s := buildStateSensor(t, server.URL, "light.a", 0.01) // 10ms
		_, err := s.Poll(context.Background())
		require.NoError(t, err)

		time.Sleep(20 * time.Millisecond)

		raw, err := s.Poll(context.Background())
		require.NoError(t, err)
		require.NotNil(t, raw)
	})

	t.Run("returns nil with no entity_ids", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "", 0)
		raw, err := s.Poll(context.Background())
		require.NoError(t, err)
		require.Nil(t, raw)
	})

	t.Run("skips failed fetches", func(t *testing.T) {
		server := newMockHAStatesServer(t, map[string]map[string]any{
			"light.a": {"entity_id": "light.a", "state": "on"},
		})
		defer server.Close()

		s := buildStateSensor(t, server.URL, "light.a,light.missing", 0.01)
		time.Sleep(20 * time.Millisecond)

		raw, err := s.Poll(context.Background())
		require.NoError(t, err)
		states, ok := raw.([]map[string]any)
		require.True(t, ok)
		require.Len(t, states, 1)
	})
}

func TestFormatState(t *testing.T) {
	t.Run("basic state", func(t *testing.T) {
		state := map[string]any{
			"entity_id":  "light.bed_light",
			"state":      "on",
			"attributes": map[string]any{"friendly_name": "Bed Light"},
		}
		require.Equal(t, "Bed Light (light.bed_light) is on", formatState(state))
	})

	t.Run("with brightness", func(t *testing.T) {
		state := map[string]any{
			"entity_id": "light.bed_light",
			"state":     "on",
			"attributes": map[string]any{
				"friendly_name": "Bed Light",
				"brightness":    127.5,
			},
		}
		require.Contains(t, formatState(state), "brightness 50%")
	})

	t.Run("with color", func(t *testing.T) {
		state := map[string]any{
			"entity_id": "light.bed_light",
			"state":     "on",
			"attributes": map[string]any{
				"friendly_name": "Bed Light",
				"color_name":    "red",
			},
		}
		require.Contains(t, formatState(state), "color red")
	})

	t.Run("with temperature", func(t *testing.T) {
		state := map[string]any{
			"entity_id":  "climate.living_room",
			"state":      "heat",
			"attributes": map[string]any{"temperature": 22.5},
		}
		require.Contains(t, formatState(state), "temperature 22.5°C")
	})

	t.Run("no friendly_name falls back to entity_id", func(t *testing.T) {
		state := map[string]any{
			"entity_id":  "light.bed_light",
			"state":      "on",
			"attributes": map[string]any{},
		}
		require.Contains(t, formatState(state), "light.bed_light (light.bed_light)")
	})
}

func TestRawToText(t *testing.T) {
	t.Run("none input returns nil", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		msg, err := s.RawToText(context.Background(), nil)
		require.NoError(t, err)
		require.Nil(t, msg)
	})

	t.Run("new state returns message", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		msg, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)
		require.NotNil(t, msg)
		require.Contains(t, msg.Message, "Smart home device updates")
	})

	t.Run("no change returns nil", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		_, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)

		msg, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)
		require.Nil(t, msg)
	})

	t.Run("detects state change", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		s.lastStates["light.a"] = "off"
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		msg, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)
		require.NotNil(t, msg)
	})

	t.Run("updates last states", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		_, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)
		require.Equal(t, "on", s.lastStates["light.a"])
	})

	t.Run("adds to messages buffer", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		_, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)
		require.Len(t, s.messages, 1)
	})

	t.Run("nil result does not add message", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		_, err := s.RawToText(context.Background(), nil)
		require.NoError(t, err)
		require.Empty(t, s.messages)
	})
}

func TestFormattedLatestBuffer(t *testing.T) {
	t.Run("empty buffer returns empty string", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		require.Empty(t, s.FormattedLatestBuffer())
	})

	t.Run("with message returns formatted string and clears buffer", func(t *testing.T) {
		s := buildStateSensor(t, "http://example.com", "light.a", 0)
		states := []map[string]any{
			{"entity_id": "light.a", "state": "on", "attributes": map[string]any{}},
		}
		_, err := s.RawToText(context.Background(), states)
		require.NoError(t, err)

		result := s.FormattedLatestBuffer()
		require.Contains(t, result, "INPUT: Home Assistant Device States")
		require.Contains(t, result, "// START")
		require.Contains(t, result, "// END")
		require.Empty(t, s.messages)
	})
}

func TestStateSensorStopIdempotent(t *testing.T) {
	s := buildStateSensor(t, "http://example.com", "light.a", 0)
	require.NotPanics(t, s.Stop)
	require.NotPanics(t, s.Stop) // calling twice should be safe
}
