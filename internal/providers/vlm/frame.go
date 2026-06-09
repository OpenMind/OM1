package vlm

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// Frame represents a single video frame with a timestamp and JPEG-encoded data.
type Frame struct {
	Timestamp time.Time
	JPEG      []byte
}

// framePayload is the JSON-serializable representation of a Frame.
type framePayload struct {
	Timestamp float64 `json:"timestamp"`
	Frame     string  `json:"frame"`
}

// MarshalJSON implements custom JSON encoding for Frame, encoding the timestamp as seconds since the Unix epoch and the JPEG data as a base64 string.
func (f Frame) MarshalJSON() ([]byte, error) {
	return json.Marshal(framePayload{
		Timestamp: float64(f.Timestamp.UnixNano()) / float64(time.Second),
		Frame:     base64.StdEncoding.EncodeToString(f.JPEG),
	})
}

// streamBase provides common lifecycle and buffering logic for video streams.
type streamBase struct {
	name       string
	log        *zap.Logger
	bufferSize int

	mu      sync.Mutex
	cancel  context.CancelFunc
	out     chan Frame
	done    chan struct{}
	running bool
	dropped atomic.Uint64
}

// start begins the capture loop in a new goroutine and returns the output channel.
func (b *streamBase) start(ctx context.Context, loop func(context.Context)) <-chan Frame {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.running {
		return b.out
	}

	bufSize := b.bufferSize
	if bufSize <= 0 {
		bufSize = 1
	}

	cctx, cancel := context.WithCancel(ctx)
	b.out = make(chan Frame, bufSize)
	b.done = make(chan struct{})
	b.cancel = cancel
	b.running = true
	out, done := b.out, b.done

	go func() {
		defer close(done)
		defer close(out)
		loop(cctx)
	}()

	b.log.Info(b.name + ": started video processing")
	return out
}

// send attempts to send the frame to the output channel without blocking.
func (b *streamBase) send(f Frame) {
	select {
	case b.out <- f:
	default:
		if n := b.dropped.Add(1); n%100 == 1 {
			b.log.Debug(b.name+": dropping frame, consumer not keeping up",
				zap.Uint64("dropped_total", n))
		}
	}
}

// Stop signals the capture goroutine to stop and waits for it to return.
func (b *streamBase) Stop() {
	b.mu.Lock()
	if !b.running {
		b.mu.Unlock()
		return
	}

	b.running = false
	cancel := b.cancel
	done := b.done
	b.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if done != nil {
		select {
		case <-done:
		case <-time.After(time.Second):
		}
	}

	b.log.Info(b.name + ": stopped video processing")
}

const (
	jpegMarker = 0xFF
	jpegSOI    = 0xD8
	jpegEOI    = 0xD9
)

// splitJPEGStream reads from r and calls onFrame for each complete JPEG frame found.
func splitJPEGStream(r io.Reader, onFrame func(frame []byte) bool) error {
	br := bufio.NewReaderSize(r, 256*1024)
	frame := make([]byte, 0, 64*1024)
	inFrame := false
	var last byte

	for {
		b, err := br.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}

		if inFrame {
			frame = append(frame, b)
			if last == jpegMarker && b == jpegEOI {
				out := make([]byte, len(frame))
				copy(out, frame)
				if !onFrame(out) {
					return nil
				}

				inFrame = false
				frame = frame[:0]
				last = 0
				continue
			}

			last = b
			continue
		}

		if last == jpegMarker && b == jpegSOI {
			inFrame = true
			frame = append(frame[:0], jpegMarker, jpegSOI)
			last = jpegSOI
			continue
		}

		last = b
	}
}

// jpegQScale maps a user-friendly quality percentage [0, 100] to ffmpeg's JPEG qscale [2, 31].
func jpegQScale(quality int) int {
	if quality < 0 {
		quality = 0
	}

	if quality > 100 {
		quality = 100
	}

	q := 31 - (quality*29+50)/100
	if q < 2 {
		q = 2
	}

	if q > 31 {
		q = 31
	}

	return q
}
