package asr

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers/tts"
	"github.com/openmind/om1/internal/ws"
)

// newSendableStream builds a transcriberStream with a websocket client whose
// send buffer is large enough that Send() never blocks or fills. The client is
// never Connect()-ed, so no goroutines start and Send simply enqueues onto the
// buffered channel — enough to exercise packaging + statistics without a server.
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
	// A single-slot buffer that we pre-fill, so the next Send fails.
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

func TestForwardChunkSendsWhenNotSpeaking(t *testing.T) {
	s := newSendableStream(t)
	c := &asrCommon{asrSensorCore: newTestSensorCore(), stream: s}
	// Ensure TTS is not "speaking" for this case.
	tts.Speaking.Store(false)

	sent := c.forwardChunk([]byte{0x01, 0x02, 0x03, 0x04}, time.UnixMilli(1_700_000_000_000))

	require.True(t, sent, "chunk must be forwarded when TTS is silent")
	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Equal(t, uint64(1), s.stats.TotalChunksSent)
}

func TestForwardChunkDropsWhileSpeakingWithoutInterrupt(t *testing.T) {
	s := newSendableStream(t)
	core := newTestSensorCore()
	core.enableTTSInterrupt = false
	c := &asrCommon{asrSensorCore: core, stream: s}

	tts.Speaking.Store(true)
	defer tts.Speaking.Store(false)

	sent := c.forwardChunk([]byte{0x01, 0x02}, time.Now())

	require.False(t, sent, "chunk must be dropped while TTS speaks and interrupt is disabled")
	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Zero(t, s.stats.TotalChunksSent)
}

func TestForwardChunkSendsWhileSpeakingWithInterrupt(t *testing.T) {
	s := newSendableStream(t)
	core := newTestSensorCore()
	core.enableTTSInterrupt = true
	c := &asrCommon{asrSensorCore: core, stream: s}

	tts.Speaking.Store(true)
	defer tts.Speaking.Store(false)

	sent := c.forwardChunk([]byte{0x01, 0x02}, time.Now())

	require.True(t, sent, "interrupt-enabled sensors keep streaming during TTS")
	s.stats.mu.RLock()
	defer s.stats.mu.RUnlock()
	require.Equal(t, uint64(1), s.stats.TotalChunksSent)
}

// TestASRRTSPDefaultURLs pins the RTSP audio source defaults to the
// video-processor's muxed session stream (:8555/live) and verifies an explicit
// URL is preserved.
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
