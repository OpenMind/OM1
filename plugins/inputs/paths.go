package inputs

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("Paths", NewPaths)
}

const (
	pathsDescriptor  = "Information about objects and walls around you, to plan your movements and avoid bumping into things."
	pathsPollPeriod  = 200 * time.Millisecond
	pathsMaxMessages = 10
)

type PathsSensor struct {
	log      *zap.Logger
	provider *providers.PathsProvider

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewPaths constructs a PathsSensor and starts its underlying PathsProvider.
func NewPaths(_ map[string]any) (inputs.Sensor, error) {
	log := logger.Get().Named("Paths")
	log.Info("initializing")

	return &PathsSensor{
		log:      log,
		provider: providers.NewPathsProvider(),
	}, nil
}

// Listen polls the paths provider at a fixed cadence and yields each non-empty
// assessment on the returned channel until ctx is cancelled.
func (s *PathsSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(pathsPollPeriod)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			raw, err := s.Poll(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				continue
			}

			select {
			case out <- raw:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// Poll returns the latest path assessment from the provider.
func (s *PathsSensor) Poll(_ context.Context) (any, error) {
	return s.provider.LidarString(), nil
}

// RawToText appends a new assessment to the message buffer, returning it as an inputs.Message for downstream processing.
func (s *PathsSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > pathsMaxMessages {
		s.messages = s.messages[len(s.messages)-pathsMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the most recent assessment in the buffer.
func (s *PathsSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: %q\n", pathsDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(pathsDescriptor, latest.Message, ts)
	s.messages = nil

	return result
}

// Stop releases the underlying provider.
func (s *PathsSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
	if s.provider != nil {
		s.provider.Stop()
	}
}
