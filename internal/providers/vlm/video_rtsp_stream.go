package vlm

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/util"
)

const (
	defaultRTSPURL     = "rtsp://localhost:8554/live"
	defaultRTSPWidth   = 480
	defaultRTSPHeight  = 640
	rtspReconnectDelay = 2 * time.Second
)

// VideoRTSPStreamConfig holds configuration options for the VideoRTSPStream.
type VideoRTSPStreamConfig struct {
	RTSPURL string

	FPS int

	Width  int
	Height int

	JPEGQuality int
	BufferSize  int
}

// VideoRTSPStream captures video frames from an RTSP source using ffmpeg and emits them as JPEG-encoded Frames.
type VideoRTSPStream struct {
	*streamBase
	cfg VideoRTSPStreamConfig
}

// NewVideoRTSPStream creates a VideoRTSPStream with the given config.
func NewVideoRTSPStream(cfg VideoRTSPStreamConfig) *VideoRTSPStream {
	if cfg.RTSPURL == "" {
		cfg.RTSPURL = defaultRTSPURL
	}

	if cfg.FPS <= 0 {
		cfg.FPS = defaultFPS
	}

	if cfg.Width <= 0 {
		cfg.Width = defaultRTSPWidth
	}

	if cfg.Height <= 0 {
		cfg.Height = defaultRTSPHeight
	}

	if cfg.JPEGQuality <= 0 {
		cfg.JPEGQuality = defaultJPEGQuality
	}

	return &VideoRTSPStream{
		streamBase: &streamBase{
			name:       "VideoRTSPStream",
			log:        logger.Get(),
			bufferSize: cfg.BufferSize,
		},
		cfg: cfg,
	}
}

// Start begins the RTSP capture loop and returns the output channel of Frames.
func (v *VideoRTSPStream) Start(ctx context.Context) <-chan Frame {
	return v.start(ctx, v.captureLoop)
}

// captureLoop manages the lifecycle of the RTSP streaming session.
func (v *VideoRTSPStream) captureLoop(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}

		if err := v.stream(ctx); err != nil && ctx.Err() == nil {
			v.log.Warn("VideoRTSPStream: error streaming video", zap.Error(err))
		}

		if ctx.Err() != nil {
			return
		}
		v.log.Info("VideoRTSPStream: reconnecting", zap.Duration("delay", rtspReconnectDelay))
		if !util.Sleep(ctx, rtspReconnectDelay) {
			return
		}
	}
}

// stream runs a single ffmpeg RTSP session and forwards decoded JPEG frames.
func (v *VideoRTSPStream) stream(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "ffmpeg", v.ffmpegArgs()...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}

	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
		v.log.Debug("VideoRTSPStream: released video capture device")
	}()

	v.log.Info("VideoRTSPStream: RTSP stream connected", zap.String("rtsp_url", v.cfg.RTSPURL))

	return splitJPEGStream(stdout, func(frame []byte) bool {
		if ctx.Err() != nil {
			return false
		}
		v.send(Frame{Timestamp: time.Now(), JPEG: frame})
		return true
	})
}

// ffmpegArgs builds the ffmpeg argument list for RTSP capture.
func (v *VideoRTSPStream) ffmpegArgs() []string {
	return []string{
		"-loglevel", "error",
		"-rtsp_transport", "tcp",
		"-i", v.cfg.RTSPURL,
		"-an",
		"-vf", fmt.Sprintf("scale=%d:%d,fps=%d", v.cfg.Width, v.cfg.Height, v.cfg.FPS),
		"-c:v", "mjpeg",
		"-qscale:v", strconv.Itoa(jpegQScale(v.cfg.JPEGQuality)),
		"-f", "image2pipe",
		"pipe:1",
	}
}
