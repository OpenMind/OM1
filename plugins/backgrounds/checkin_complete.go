package backgrounds

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
)

func init() {
	bg.Register("CheckinComplete", NewCheckinComplete)
}

type checkinCompleteConfig struct {
	FaceBaseURL    string  `json:"face_http_base_url"`
	FaceRecentSec  float64 `json:"face_recent_sec"`
	FaceMinArea    float64 `json:"min_face_area"`
	PollSec        float64 `json:"poll_interval_sec"`
	ScanIOKey      string  `json:"scan_io_key"`
	GracePeriodSec float64 `json:"grace_period_sec"`
}

// CheckinComplete publishes checkin_complete:true to ModeContext once a
// successful QR scan has been recorded AND the guest's face is no longer
// visible (or too small).
type CheckinComplete struct {
	log         *zap.Logger
	face        *providers.FacePresenceProvider
	period      time.Duration
	scanIOKey   string
	gracePeriod time.Duration
	minArea     float64
	scanSeen    bool
	scanTime    time.Time
}

func NewCheckinComplete(configMap map[string]any) (bg.Background, error) {
	var cfg checkinCompleteConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.FaceBaseURL == "" {
		cfg.FaceBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.FaceRecentSec <= 0 {
		cfg.FaceRecentSec = 1.0
	}
	if cfg.FaceMinArea <= 0 {
		cfg.FaceMinArea = 3000
	}
	if cfg.PollSec <= 0 {
		cfg.PollSec = 1.0
	}
	if cfg.ScanIOKey == "" {
		cfg.ScanIOKey = "QRScannerRTSP"
	}
	if cfg.GracePeriodSec <= 0 {
		cfg.GracePeriodSec = 5.0
	}

	log := logger.Get().Named("CheckinComplete")

	face := providers.NewFacePresenceProvider(providers.FacePresenceConfig{
		BaseURL:   cfg.FaceBaseURL,
		RecentSec: cfg.FaceRecentSec,
		Timeout:   2 * time.Second,
	})

	log.Info("initialized",
		zap.String("scan_io_key", cfg.ScanIOKey),
		zap.Float64("grace_period_sec", cfg.GracePeriodSec),
	)

	return &CheckinComplete{
		log:         log,
		face:        face,
		period:      time.Duration(cfg.PollSec * float64(time.Second)),
		scanIOKey:   cfg.ScanIOKey,
		gracePeriod: time.Duration(cfg.GracePeriodSec * float64(time.Second)),
		minArea:     cfg.FaceMinArea,
	}, nil
}

func (c *CheckinComplete) Run(ctx context.Context) {
	// Step 1: check if a successful scan has been recorded.
	if !c.scanSeen {
		if in := providers.IO().GetInput(c.scanIOKey); in != nil {
			if strings.HasPrefix(in.Input, "qr_scan: name=") {
				c.scanSeen = true
				c.scanTime = in.Timestamp
				c.log.Info("successful scan detected", zap.String("value", in.Input))
			}
		}
	}

	if !c.scanSeen {
		util.Sleep(ctx, c.period)
		return
	}

	// Step 2: wait a grace period after the scan before checking for face
	// departure, so the greeting has time to play.
	if time.Since(c.scanTime) < c.gracePeriod {
		util.Sleep(ctx, c.period)
		return
	}

	// Step 3: check if the face is gone.
	snap, err := c.face.FetchSnapshot(ctx)
	if err != nil {
		if ctx.Err() == nil {
			c.log.Warn("failed to fetch face snapshot", zap.Error(err))
		}
		util.Sleep(ctx, c.period)
		return
	}

	var totalFaces int
	for _, face := range snap.Faces {
		if float64(face.Area) >= c.minArea {
			totalFaces++
		}
	}
	if totalFaces > 0 {
		util.Sleep(ctx, c.period)
		return
	}

	c.log.Info("guest departed after successful scan, triggering transition")
	providers.ModeContext().Publish(map[string]any{"checkin_complete": true})
	c.scanSeen = false

	util.Sleep(ctx, c.period)
}

func (c *CheckinComplete) Stop() {
	c.log.Info("stopping")
}
