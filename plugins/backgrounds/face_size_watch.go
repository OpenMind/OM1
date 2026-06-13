package backgrounds

import (
	"context"
	"encoding/json"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
)

func init() {
	bg.Register("FaceSizeWatch", NewFaceSizeWatch)
}

type faceSizeWatchConfig struct {
	BaseURL     string  `json:"face_http_base_url"`
	RecentSec   float64 `json:"face_recent_sec"`
	PollSec     float64 `json:"face_poll_interval_sec"`
	MinFaceArea float64 `json:"min_face_area"`
}

type FaceSizeWatch struct {
	log      *zap.Logger
	provider *providers.FacePresenceProvider
	period   time.Duration
	minArea  float64
}

func NewFaceSizeWatch(configMap map[string]any) (bg.Background, error) {
	var cfg faceSizeWatchConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://127.0.0.1:6793"
	}
	if cfg.RecentSec <= 0 {
		cfg.RecentSec = 1.0
	}
	if cfg.PollSec <= 0 {
		cfg.PollSec = 0.5
	}
	if cfg.MinFaceArea <= 0 {
		cfg.MinFaceArea = 3000
	}

	log := logger.Get().Named("FaceSizeWatch")

	provider := providers.NewFacePresenceProvider(providers.FacePresenceConfig{
		BaseURL:   cfg.BaseURL,
		RecentSec: cfg.RecentSec,
		Timeout:   2 * time.Second,
	})

	log.Info("initialized",
		zap.String("base_url", cfg.BaseURL),
		zap.Float64("min_face_area", cfg.MinFaceArea),
		zap.Float64("poll_sec", cfg.PollSec),
	)

	return &FaceSizeWatch{
		log:      log,
		provider: provider,
		period:   time.Duration(cfg.PollSec * float64(time.Second)),
		minArea:  cfg.MinFaceArea,
	}, nil
}

func (f *FaceSizeWatch) Run(ctx context.Context) {
	snap, err := f.provider.FetchSnapshot(ctx)
	if err != nil {
		if ctx.Err() == nil {
			f.log.Warn("failed to fetch snapshot", zap.Error(err))
		}
		util.Sleep(ctx, f.period)
		return
	}

	var totalFaces int
	for _, face := range snap.Faces {
		if float64(face.Area) >= f.minArea {
			totalFaces++
		}
	}
	if totalFaces > 0 {
		f.log.Info("face close enough, triggering transition",
			zap.Int("faces", totalFaces),
			zap.String("closest", snap.ClosestName),
		)
		providers.ModeContext().Publish(map[string]any{"face_close_enough": true})
	}

	util.Sleep(ctx, f.period)
}

func (f *FaceSizeWatch) Stop() {
	f.log.Info("stopping")
}
