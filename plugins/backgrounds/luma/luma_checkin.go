package luma

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/luma"
	"github.com/openmind/om1/internal/util"
)

func init() {
	bg.Register("LumaCheckin", LumaCheckin)
}

type checkinCompleteConfig struct {
	FaceBaseURL    string  `json:"face_http_base_url"`
	FaceRecentSec  float64 `json:"face_recent_sec"`
	FaceMinArea    float64 `json:"min_face_area"`
	PollSec        float64 `json:"poll_interval_sec"`
	GracePeriodSec float64 `json:"grace_period_sec"`
}

type CheckinComplete struct {
	log         *zap.Logger
	face        *providers.FacePresenceProvider
	period      time.Duration
	gracePeriod time.Duration
	minArea     float64
	lastHandled time.Time // timestamp of the scan we've already acted on
}

func LumaCheckin(configMap map[string]any) (bg.Background, error) {
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
		zap.Float64("grace_period_sec", cfg.GracePeriodSec),
	)

	return &CheckinComplete{
		log:         log,
		face:        face,
		period:      time.Duration(cfg.PollSec * float64(time.Second)),
		gracePeriod: time.Duration(cfg.GracePeriodSec * float64(time.Second)),
		minArea:     cfg.FaceMinArea,
		lastHandled: time.Now(),
	}, nil
}

func (c *CheckinComplete) Run(ctx context.Context) {
	// Step 1: pick up the latest successful check-in. Skip when there is none yet
	// or when we've already acted on it.
	checkin := luma.LastCheckIn()
	if checkin == nil || !checkin.Time.After(c.lastHandled) {
		util.Sleep(ctx, c.period)
		return
	}

	// Step 2: wait a grace period after the check-in before checking for face
	// departure, so the greeting has time to play.
	if time.Since(checkin.Time) < c.gracePeriod {
		util.Sleep(ctx, c.period)
		return
	}

	// Read the primary guest's track_id stored by FaceSizeWatch.
	var primaryTrackID int
	if in := providers.IO().GetInput("PrimaryGuestTrackID"); in != nil && in.Input != "" {
		primaryTrackID, _ = strconv.Atoi(in.Input)
	}

	// Step 3: check if the primary guest's face is gone.
	snap, err := c.face.FetchSnapshot(ctx)
	if err != nil {
		if ctx.Err() == nil {
			c.log.Warn("failed to fetch face snapshot", zap.Error(err))
		}
		util.Sleep(ctx, c.period)
		return
	}

	for _, face := range snap.Faces {
		if primaryTrackID > 0 && face.TrackID == primaryTrackID && float64(face.Area) >= c.minArea {
			util.Sleep(ctx, c.period)
			return
		}
	}
	// Fallback: if no track_id match, check if any large face remains.
	if primaryTrackID == 0 {
		for _, face := range snap.Faces {
			if float64(face.Area) >= c.minArea {
				util.Sleep(ctx, c.period)
				return
			}
		}
	}

	c.log.Info("guest departed after successful check-in, triggering transition", zap.String("name", checkin.Name))
	providers.ModeContext().Publish(map[string]any{"checkin_complete": true})
	c.lastHandled = checkin.Time

	util.Sleep(ctx, c.period)
}

func (c *CheckinComplete) Stop() {
	c.log.Info("stopping")
}
