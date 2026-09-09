package asr

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/ws"
)

func newSendableStream(t *testing.T) *transcriberStream {
	t.Helper()
	s := newTestElevenLabsStream(make(chan string, 1))
	s.wsClient = ws.New(
		ws.Config{URL: "ws://127.0.0.1:0", SendBufferSize: 8},
		zap.NewNop(),
		nil,
	)
	return s
}

func TestSendChunkAtSuccess(t *testing.T) {
	s := newSendableStream(t)

	pcm := []byte{0x01, 0x02, 0x03, 0x04}
	capture := time.UnixMilli(1_700_000_000_000)

	s.sendChunkAt(pcm, capture)

	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Equal(t, uint64(1), s.stats.TotalChunksSent)
	require.Greater(t, s.stats.TotalBytesSent, uint64(len(pcm)),
		"bytes sent must include the JSON header, not just the PCM")
	require.Zero(t, s.stats.FailedChunks)
	require.False(t, s.stats.LastSendTime.IsZero(), "a successful send must record LastSendTime")
}

func TestSendChunkAtSendError(t *testing.T) {
	s := newTestElevenLabsStream(make(chan string, 1))
	s.wsClient = ws.New(
		ws.Config{URL: "ws://127.0.0.1:0", SendBufferSize: 1},
		zap.NewNop(),
		nil,
	)
	require.NoError(t, s.wsClient.Send([]byte{0x00}), "prime the send buffer to capacity")

	s.sendChunkAt([]byte{0x01, 0x02}, time.Now())

	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Equal(t, uint64(1), s.stats.FailedChunks, "a full send buffer must count as a failed chunk")
	require.Zero(t, s.stats.TotalChunksSent)
}

func TestASRCommonSendChunkAtDelegates(t *testing.T) {
	s := newSendableStream(t)
	c := &asrCommon{asrSensorCore: newTestSensorCore(), stream: s}

	c.sendChunkAt([]byte{0x01, 0x02, 0x03, 0x04}, time.UnixMilli(1_700_000_000_000))

	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Equal(t, uint64(1), s.stats.TotalChunksSent,
		"asrCommon.sendChunkAt must forward to the underlying stream")
}

func TestASRRTSPDefaultURLs(t *testing.T) {
	const wantDefault = "rtsp://localhost:8555/live"

	t.Run("google", func(t *testing.T) {
		snr, err := NewGoogleASRRTSP(map[string]any{"api_key": "k"})
		require.NoError(t, err)
		require.Equal(t, wantDefault, snr.(*GoogleASRRTSPSensor).cfg.RTSPURL)
	})

	t.Run("riva", func(t *testing.T) {
		snr, err := NewRivaASRRTSP(map[string]any{})
		require.NoError(t, err)
		require.Equal(t, wantDefault, snr.(*RivaASRRTSPSensor).cfg.RTSPURL)
	})

	t.Run("elevenlabs", func(t *testing.T) {
		snr, err := NewElevenLabsASRRTSP(map[string]any{"api_key": "k"})
		require.NoError(t, err)
		require.Equal(t, wantDefault, snr.(*ElevenLabsASRRTSPSensor).cfg.RTSPURL)
	})

	t.Run("explicit url preserved", func(t *testing.T) {
		snr, err := NewGoogleASRRTSP(map[string]any{
			"api_key":  "k",
			"rtsp_url": "rtsp://example.test/custom",
		})
		require.NoError(t, err)
		require.Equal(t, "rtsp://example.test/custom", snr.(*GoogleASRRTSPSensor).cfg.RTSPURL)
	})
}
