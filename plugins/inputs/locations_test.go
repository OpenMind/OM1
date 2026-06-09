package inputs

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

func newTestLocationsSensor() *LocationsSensor {
	return &LocationsSensor{log: zap.NewNop()}
}

func TestLocationsRawToText(t *testing.T) {
	s := newTestLocationsSensor()
	ctx := context.Background()

	msg, err := s.RawToText(ctx, 123)
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = s.RawToText(ctx, "")
	require.NoError(t, err)
	require.Nil(t, msg)

	msg, err = s.RawToText(ctx, "kitchen")
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, "kitchen", msg.Message)
	require.Len(t, s.messages, 1)
}

func TestLocationsRawToTextCapsBuffer(t *testing.T) {
	s := newTestLocationsSensor()
	ctx := context.Background()

	for i := 0; i < locationsMaxMessages+5; i++ {
		_, err := s.RawToText(ctx, "loc")
		require.NoError(t, err)
	}

	require.Len(t, s.messages, locationsMaxMessages, "buffer is capped at locationsMaxMessages")
}

func TestLocationsFormattedLatestBuffer(t *testing.T) {
	s := newTestLocationsSensor()

	require.Equal(t, "", s.FormattedLatestBuffer())

	_, err := s.RawToText(context.Background(), "Kitchen (x:1.50 y:-2.00)")
	require.NoError(t, err)

	out := s.FormattedLatestBuffer()
	require.Contains(t, out, locationsDescriptor)
	require.Contains(t, out, "Kitchen (x:1.50 y:-2.00)")

	require.Empty(t, s.messages)
	require.Equal(t, "", s.FormattedLatestBuffer())
}

func TestLocationsPollFormatsLocations(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"Kitchen": {"name": "Kitchen", "pose": {"position": {"x": 1.5, "y": -2.0, "z": 0}}},
			"sofa": {"name": "sofa"}
		}`))
	}))
	defer srv.Close()

	provider := providers.NewLocationsProvider(srv.URL, "", time.Second, time.Hour)
	provider.Start()
	defer provider.Stop()

	s := &LocationsSensor{log: zap.NewNop(), provider: provider}

	var out string
	require.Eventually(t, func() bool {
		raw, err := s.Poll(context.Background())
		require.NoError(t, err)
		out, _ = raw.(string)
		return out != ""
	}, time.Second, 10*time.Millisecond)

	lines := strings.Split(out, "\n")
	require.Equal(t, []string{
		"Kitchen (x:1.50 y:-2.00)",
		"sofa",
	}, lines, "lines are sorted and formatted per pose presence")
}

func TestLocationsPollEmpty(t *testing.T) {
	provider := providers.NewLocationsProvider("http://127.0.0.1:0", "", time.Second, time.Hour)
	s := &LocationsSensor{log: zap.NewNop(), provider: provider}

	raw, err := s.Poll(context.Background())
	require.NoError(t, err)
	require.Equal(t, "", raw, "no cached locations yields an empty string")
}

func TestLocationsStopIdempotent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	provider := providers.NewLocationsProvider(srv.URL, "", time.Second, time.Hour)
	provider.Start()
	s := &LocationsSensor{log: zap.NewNop(), provider: provider}

	s.Stop()
	require.True(t, s.stopped)
	s.Stop()
}
