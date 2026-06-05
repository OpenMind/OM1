package vlm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func jpeg(payload ...byte) []byte {
	out := []byte{jpegMarker, jpegSOI}
	out = append(out, payload...)
	return append(out, jpegMarker, jpegEOI)
}

func TestSplitJPEGStream(t *testing.T) {
	f1 := jpeg(0x01, 0x02, 0x03)
	f2 := jpeg(0xAA, 0xBB)

	stream := append([]byte{0x00, 0xFF, 0x10}, f1...)
	stream = append(stream, f2...)

	var got [][]byte
	err := splitJPEGStream(bytes.NewReader(stream), func(frame []byte) bool {
		got = append(got, frame)
		return true
	})

	require.NoError(t, err)
	require.Len(t, got, 2)
	assert.Equal(t, f1, got[0])
	assert.Equal(t, f2, got[1])
}

func TestSplitJPEGStreamStopsWhenCallbackReturnsFalse(t *testing.T) {
	stream := append(jpeg(0x01), jpeg(0x02)...)

	count := 0
	err := splitJPEGStream(bytes.NewReader(stream), func(frame []byte) bool {
		count++
		return false
	})

	require.NoError(t, err)
	assert.Equal(t, 1, count)
}

func TestJPEGQScale(t *testing.T) {
	assert.Equal(t, 2, jpegQScale(100))
	assert.Equal(t, 31, jpegQScale(0))
	assert.Equal(t, 2, jpegQScale(150))
	assert.Equal(t, 31, jpegQScale(-10))

	prev := jpegQScale(0)
	for q := 1; q <= 100; q++ {
		cur := jpegQScale(q)
		assert.LessOrEqual(t, cur, prev, "qscale should not increase as quality rises (q=%d)", q)
		assert.GreaterOrEqual(t, cur, 2)
		assert.LessOrEqual(t, cur, 31)
		prev = cur
	}
}

func TestFrameMarshalJSON(t *testing.T) {
	f := Frame{
		Timestamp: time.Unix(1700000000, 500000000),
		JPEG:      []byte{0xDE, 0xAD, 0xBE, 0xEF},
	}

	data, err := json.Marshal(f)
	require.NoError(t, err)

	var payload framePayload
	require.NoError(t, json.Unmarshal(data, &payload))
	assert.InDelta(t, 1700000000.5, payload.Timestamp, 1e-6)

	decoded, err := base64.StdEncoding.DecodeString(payload.Frame)
	require.NoError(t, err)
	assert.Equal(t, f.JPEG, decoded)
}

func TestStreamBaseSendDropsWhenFull(t *testing.T) {
	b := &streamBase{name: "test", log: zap.NewNop()}
	b.out = make(chan Frame, 2)

	b.send(Frame{JPEG: []byte{1}})
	b.send(Frame{JPEG: []byte{2}})
	b.send(Frame{JPEG: []byte{3}})

	assert.Equal(t, uint64(1), b.dropped.Load())
	assert.Len(t, b.out, 2)
}

func TestStreamBaseLifecycle(t *testing.T) {
	b := &streamBase{name: "test", log: zap.NewNop(), bufferSize: 4}

	started := make(chan struct{})
	frames := b.start(context.Background(), func(ctx context.Context) {
		close(started)
		b.send(Frame{JPEG: []byte{0x42}})
		<-ctx.Done()
	})

	<-started
	f := <-frames
	assert.Equal(t, []byte{0x42}, f.JPEG)

	assert.Equal(t, frames, b.start(context.Background(), func(context.Context) {
		t.Error("loop should not be started again while running")
	}))

	b.Stop()

	for range frames {
	}
}

func TestStopWithoutStart(t *testing.T) {
	NewVideoStream(VideoStreamConfig{}).Stop()
	NewVideoRTSPStream(VideoRTSPStreamConfig{}).Stop()
}

func TestParseAVFoundationDevices(t *testing.T) {
	output := `[AVFoundation indev @ 0x123] AVFoundation video devices:
[AVFoundation indev @ 0x123] [0] FaceTime HD Camera
[AVFoundation indev @ 0x123] [1] Capture screen 0
[AVFoundation indev @ 0x123] AVFoundation audio devices:
[AVFoundation indev @ 0x123] [0] MacBook Pro Microphone`

	devices := parseAVFoundationDevices(output)
	require.Len(t, devices, 2)
	assert.Equal(t, VideoDevice{Index: 0, Name: "FaceTime HD Camera"}, devices[0])
	assert.Equal(t, VideoDevice{Index: 1, Name: "Capture screen 0"}, devices[1])
}

func TestNewVideoStreamDefaults(t *testing.T) {
	v := NewVideoStream(VideoStreamConfig{})
	assert.Equal(t, defaultFPS, v.cfg.FPS)
	assert.Equal(t, defaultWidth, v.cfg.Width)
	assert.Equal(t, defaultHeight, v.cfg.Height)
	assert.Equal(t, defaultJPEGQuality, v.cfg.JPEGQuality)
}

func TestNewVideoRTSPStreamDefaults(t *testing.T) {
	v := NewVideoRTSPStream(VideoRTSPStreamConfig{})
	assert.Equal(t, defaultRTSPURL, v.cfg.RTSPURL)
	assert.Equal(t, defaultRTSPDecodeFormat, v.cfg.DecodeFormat)
	assert.Equal(t, defaultFPS, v.cfg.FPS)
	assert.Equal(t, defaultRTSPWidth, v.cfg.Width)
	assert.Equal(t, defaultRTSPHeight, v.cfg.Height)
}
