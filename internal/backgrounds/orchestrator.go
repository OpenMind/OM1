package backgrounds

import (
	"context"
	"sync"

	"go.uber.org/zap"
)

// Orchestrator runs Background tasks in separate goroutines.
type Orchestrator struct {
	backgrounds []Background
	log         *zap.Logger
}

// NewOrchestrator creates a new Orchestrator with the given Background tasks and logger.
func NewOrchestrator(backgroundList []Background, log *zap.Logger) *Orchestrator {
	return &Orchestrator{backgrounds: backgroundList, log: log}
}

// Start launches one goroutine per background and returns a channel that is
// closed when all goroutines have finished.
func (o *Orchestrator) Start(ctx context.Context) <-chan struct{} {
	done := make(chan struct{})
	var wg sync.WaitGroup

	for _, background := range o.backgrounds {
		wg.Add(1)
		go func(background Background) {
			defer wg.Done()
			o.runLoop(ctx, background)
		}(background)
	}

	go func() {
		wg.Wait()
		close(done)
	}()
	return done
}

// runLoop continuously calls background.Run(ctx) until the context is cancelled,
// recovering from panics so one failure doesn't stop the loop.
func (o *Orchestrator) runLoop(ctx context.Context, background Background) {
	defer background.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		o.safeRun(ctx, background)
	}
}

// safeRun calls background.Run(ctx) and recovers from any panics, logging the error.
func (o *Orchestrator) safeRun(ctx context.Context, background Background) {
	defer func() {
		if r := recover(); r != nil {
			o.log.Error("background panicked", zap.Any("panic", r))
		}
	}()
	background.Run(ctx)
}
