package vlm

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os/exec"
	"runtime"
	"strconv"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/util"
)

const (
	defaultWidth       = 640
	defaultHeight      = 480
	defaultFPS         = 30
	defaultJPEGQuality = 30
	cameraRetryDelay   = 2 * time.Second
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

	for {
		if ctx.Err() != nil {
			return
		}

		if err := v.stream(ctx, cam); err != nil && ctx.Err() == nil {
			v.log.Error("VideoStream: error streaming video", zap.Error(err))
		}

		if ctx.Err() != nil {
			return
		}

		v.log.Warn("VideoStream: capture ended, restarting camera", zap.Duration("delay", cameraRetryDelay))
		if !util.Sleep(ctx, cameraRetryDelay) {
			return
		}
	}
}

// stream runs a single ffmpeg capture session and forwards decoded JPEG frames
func (v *VideoStream) stream(ctx context.Context, cam string) error {
	cmd := exec.CommandContext(ctx, "ffmpeg", v.ffmpegArgs(cam)...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}

	go v.logFFmpegStderr(stderr)

	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}

		_ = cmd.Wait()
		v.log.Info("VideoStream: released video capture device")
	}()

	return splitJPEGStream(stdout, func(frame []byte) bool {
		if ctx.Err() != nil {
			return false
		}

		v.send(Frame{Timestamp: time.Now(), JPEG: frame})
		return true
	})
}

func (v *VideoStream) logFFmpegStderr(r io.ReadCloser) {
	defer r.Close()
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		v.log.Warn("VideoStream: ffmpeg", zap.String("stderr", scanner.Text()))
	}
	if err := scanner.Err(); err != nil {
		v.log.Warn("VideoStream: ffmpeg stderr read error", zap.Error(err))
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
	if runtime.GOOS == "darwin" {
		// avfoundation pairs framerate to the chosen video size, so only emit
		// -framerate when -video_size is also set; otherwise ffmpeg picks a
		// default size whose mode list may not include the requested rate.
		// Pixel format is left unset so avfoundation picks the camera's native
		// format (e.g. nv12 on Apple Silicon / Continuity cameras).
		args := []string{
			"-loglevel", "error",
			"-f", "avfoundation",
		}
		if v.cfg.Width > 0 && v.cfg.Height > 0 {
			args = append(args, "-video_size", fmt.Sprintf("%dx%d", v.cfg.Width, v.cfg.Height))
			if v.cfg.FPS > 0 {
				args = append(args, "-framerate", strconv.Itoa(v.cfg.FPS))
			}
		}
		return append(args,
			"-i", cam,
			"-an",
			"-c:v", "mjpeg",
			"-qscale:v", strconv.Itoa(jpegQScale(v.cfg.JPEGQuality)),
			"-f", "image2pipe",
			"pipe:1",
		)
	}

	args := []string{
		"-loglevel", "error",
		"-f", "v4l2",
	}
	if v.cfg.FPS > 0 {
		args = append(args, "-framerate", strconv.Itoa(v.cfg.FPS))
	}
	if v.cfg.Width > 0 && v.cfg.Height > 0 {
		args = append(args, "-video_size", fmt.Sprintf("%dx%d", v.cfg.Width, v.cfg.Height))
	}
	return append(args,
		"-i", cam,
		"-an",
		"-c:v", "mjpeg",
		"-qscale:v", strconv.Itoa(jpegQScale(v.cfg.JPEGQuality)),
		"-f", "image2pipe",
		"pipe:1",
	)
}
