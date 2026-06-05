package vlm

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
)

const (
	defaultFPS         = 30
	defaultWidth       = 640
	defaultHeight      = 480
	defaultJPEGQuality = 70
)

type VideoStreamConfig struct {
	DeviceIndex int

	FPS int

	Width  int
	Height int

	JPEGQuality int
	BufferSize  int
}

type VideoStream struct {
	*streamBase
	cfg VideoStreamConfig
}

func NewVideoStream(cfg VideoStreamConfig) *VideoStream {
	if cfg.FPS <= 0 {
		cfg.FPS = defaultFPS
	}

	if cfg.Width <= 0 {
		cfg.Width = defaultWidth
	}

	if cfg.Height <= 0 {
		cfg.Height = defaultHeight
	}

	if cfg.JPEGQuality <= 0 {
		cfg.JPEGQuality = defaultJPEGQuality
	}

	return &VideoStream{
		streamBase: &streamBase{
			name:       "VideoStream",
			log:        logger.Get(),
			bufferSize: cfg.BufferSize,
		},
		cfg: cfg,
	}
}

func (v *VideoStream) Start(ctx context.Context) <-chan Frame {
	return v.start(ctx, v.run)
}

func (v *VideoStream) run(ctx context.Context) {
	cam := v.cameraInput()
	v.log.Info("VideoStream: using camera", zap.String("camera", cam))

	cmd := exec.CommandContext(ctx, "ffmpeg", v.ffmpegArgs(cam)...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		v.log.Error("VideoStream: stdout pipe failed", zap.Error(err))
		return
	}

	if err := cmd.Start(); err != nil {
		v.log.Error("VideoStream: failed to start ffmpeg", zap.Error(err))
		return
	}

	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}

		_ = cmd.Wait()
		v.log.Info("VideoStream: released video capture device")
	}()

	err = splitJPEGStream(stdout, func(frame []byte) bool {
		if ctx.Err() != nil {
			return false
		}

		v.send(Frame{Timestamp: time.Now(), JPEG: frame})
		return true
	})

	if err != nil && ctx.Err() == nil {
		v.log.Error("VideoStream: error streaming video", zap.Error(err))
	}
}

// cameraInput returns the ffmpeg input specifier for the configured camera device.
func (v *VideoStream) cameraInput() string {
	if runtime.GOOS == "darwin" {
		return strconv.Itoa(v.cfg.DeviceIndex)
	}

	return fmt.Sprintf("/dev/video%d", v.cfg.DeviceIndex)
}

// ffmpegArgs constructs the ffmpeg command-line arguments for the configured stream.
func (v *VideoStream) ffmpegArgs(cam string) []string {
	inputFormat := "v4l2"
	if runtime.GOOS == "darwin" {
		inputFormat = "avfoundation"
	}

	return []string{
		"-loglevel", "error",
		"-f", inputFormat,
		"-framerate", strconv.Itoa(v.cfg.FPS),
		"-video_size", fmt.Sprintf("%dx%d", v.cfg.Width, v.cfg.Height),
		"-i", cam,
		"-an",
		"-c:v", "mjpeg",
		"-qscale:v", strconv.Itoa(jpegQScale(v.cfg.JPEGQuality)),
		"-f", "image2pipe",
		"pipe:1",
	}
}
