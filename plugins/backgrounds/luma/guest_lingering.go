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
	bg.Register("GuestLingering", NewGuestLingering)
}

type guestLingeringConfig struct {
	FaceBaseURL    string  `json:"face_http_base_url"`
	FaceRecentSec  float64 `json:"face_recent_sec"`
	MinFaceArea    float64 `json:"min_face_area"`
	PollSec        float64 `json:"poll_interval_sec"`
	GracePeriodSec float64 `json:"grace_period_sec"`
}

type GuestLingering struct {
	log         *zap.Logger
	face        *providers.FacePresenceProvider
	period      time.Duration
	gracePeriod time.Duration
	minArea     float64
	lastHandled time.Time
}

func NewGuestLingering(configMap map[string]any) (bg.Background, error) {
	var cfg guestLingeringConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.FaceBaseURL == "" {
		cfg.FaceBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.FaceRecentSec <= 0 {
		cfg.FaceRecentSec = 1.0
	}
	if cfg.MinFaceArea <= 0 {
		cfg.MinFaceArea = 3000
	}
	if cfg.PollSec <= 0 {
		cfg.PollSec = 1.0
	}
	if cfg.GracePeriodSec <= 0 {
		cfg.GracePeriodSec = 5.0
	}

	log := logger.Get().Named("GuestLingering")

	face := providers.NewFacePresenceProvider(providers.FacePresenceConfig{
		BaseURL:   cfg.FaceBaseURL,
		RecentSec: cfg.FaceRecentSec,
		Timeout:   2 * time.Second,
	})

	log.Info("initialized",
		zap.Float64("min_face_area", cfg.MinFaceArea),
		zap.Float64("grace_period_sec", cfg.GracePeriodSec),
	)

	return &GuestLingering{
		log:         log,
		face:        face,
		period:      time.Duration(cfg.PollSec * float64(time.Second)),
		gracePeriod: time.Duration(cfg.GracePeriodSec * float64(time.Second)),
		minArea:     cfg.MinFaceArea,
		lastHandled: time.Now(),
	}, nil
}

func (g *GuestLingering) Run(ctx context.Context) {
	checkin := luma.LastCheckIn()
	if checkin == nil || !checkin.Time.After(g.lastHandled) {
		util.Sleep(ctx, g.period)
		return
	}

	if time.Since(checkin.Time) < g.gracePeriod {
		util.Sleep(ctx, g.period)
		return
	}

	// Read the track_id of the primary guest stored by FaceSizeWatch.
	var primaryTrackID int
	if in := providers.IO().GetInput("PrimaryGuestTrackID"); in != nil && in.Input != "" {
		primaryTrackID, _ = strconv.Atoi(in.Input)
	}

	snap, err := g.face.FetchSnapshot(ctx)
	if err != nil {
		if ctx.Err() == nil {
			g.log.Warn("failed to fetch face snapshot", zap.Error(err))
		}
		util.Sleep(ctx, g.period)
		return
	}

	// Check if the primary guest's face is still present (by track_id or any large face as fallback).
	var guestPresent bool
	for _, face := range snap.Faces {
		if float64(face.Area) < g.minArea {
			continue
		}
		if primaryTrackID > 0 && face.TrackID == primaryTrackID {
			guestPresent = true
			break
		}
		if primaryTrackID == 0 {
			guestPresent = true
			break
		}
	}

	if guestPresent {
		g.log.Info("guest lingering after check-in",
			zap.String("name", checkin.Name),
			zap.Int("primary_track_id", primaryTrackID),
		)
		providers.IO().AddInput("CheckinStatus",
			"checkin_status: guest_lingering name="+checkin.Name,
			time.Now(),
		)
	} else {
		providers.IO().AddInput("CheckinStatus", "", time.Now())
		g.lastHandled = checkin.Time
	}

	util.Sleep(ctx, g.period)
}

func (g *GuestLingering) Stop() {
	g.log.Info("stopping")
}
