package autonomy

import (
	"context"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

// guardWatcher is a background poller for the face-presence service.
type guardWatcher struct {
	log      *zap.Logger
	provider *providers.FacePresenceProvider
	cancel   context.CancelFunc

	unknown atomic.Int64
}

// newGuardWatcher constructs and starts a guardWatcher with the given configuration.
func newGuardWatcher(baseURL string, log *zap.Logger) *guardWatcher {
	ctx, cancel := context.WithCancel(context.Background())

	g := &guardWatcher{
		log:      log,
		provider: providers.NewFacePresenceProvider(providers.FacePresenceConfig{BaseURL: baseURL}),
		cancel:   cancel,
	}

	go g.run(ctx)
	return g
}

func (g *guardWatcher) run(ctx context.Context) {
	ticker := time.NewTicker(guardPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			snapshot, err := g.provider.FetchSnapshot(ctx)
			if err != nil {
				g.log.Debug("face presence poll failed", zap.Error(err))
				continue
			}
			g.unknown.Store(int64(snapshot.UnknownFaces))
		}
	}
}

// unknownFaces returns the most recently observed unknown face count.
func (g *guardWatcher) unknownFaces() int {
	return int(g.unknown.Load())
}

// stop terminates the background poller.
func (g *guardWatcher) stop() {
	if g.cancel != nil {
		g.cancel()
		g.cancel = nil
	}
}
