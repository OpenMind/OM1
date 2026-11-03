package inputs

import (
	"context"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// worldState is the latest text a world sensor produced and when.
type worldState struct {
	text string
	at   time.Time
}

// Orchestrator manages multiple Sensors.
type Orchestrator struct {
	sensors []Sensor
	tickNow chan struct{}
	log     *zap.Logger

	worldMu sync.Mutex
	world   map[int]worldState // latest reading per world sensor, by index
}

// NewOrchestrator creates a new Orchestrator with the given sensors and logger.
func NewOrchestrator(sensors []Sensor, log *zap.Logger) *Orchestrator {
	return &Orchestrator{
		sensors: sensors,
		tickNow: make(chan struct{}, 1),
		log:     log,
		world:   make(map[int]worldState),
	}
}

// TickNow returns a channel that is sent a signal whenever any sensor receives new input.
func (o *Orchestrator) TickNow() <-chan struct{} {
	return o.tickNow
}

// Start launches one goroutine per sensor and returns a channel that is closed when all goroutines have finished.
func (o *Orchestrator) Start(ctx context.Context) <-chan struct{} {
	done := make(chan struct{})
	var wg sync.WaitGroup

	for i, sensor := range o.sensors {
		wg.Add(1)
		go func(sensorIndex int, sensor Sensor) {
			defer wg.Done()
			o.runSensor(ctx, sensorIndex, sensor)
		}(i, sensor)
	}

	go func() {
		wg.Wait()
		close(done)
	}()
	return done
}

// runSensor listens for readings from a sensor, converts them to text, and signals the orchestrator to tick when new input is received.
func (o *Orchestrator) runSensor(ctx context.Context, sensorIndex int, sensor Sensor) {
	readings, err := sensor.Listen(ctx)
	if err != nil {
		o.log.Warn("sensor listen failed", zap.Int("sensorIndex", sensorIndex), zap.Error(err))
		return
	}
	for {
		select {
		case rawReading, ok := <-readings:
			if !ok {
				return
			}

			message, err := sensor.RawToText(ctx, rawReading)
			if err != nil || message == nil {
				continue
			}

			if !triggersTick(sensor) {
				continue
			}

			select {
			case o.tickNow <- struct{}{}:
			default:
			}
		case <-ctx.Done():
			sensor.Stop()
			return
		}
	}
}

// Observations returns this tick's observations with their roles. Conversation and event sensors contribute
// only when they have new text; world sensors contribute their latest text every tick, marked with its age,
// so the surroundings stay in the prompt even when no new reading arrived.
func (o *Orchestrator) Observations() []Observation {
	now := time.Now()
	out := make([]Observation, 0, len(o.sensors))
	for i, sensor := range o.sensors {
		text := strings.TrimSpace(sensor.FormattedLatestBuffer())
		role := RoleOf(sensor)
		if role != RoleWorld {
			if text != "" {
				out = append(out, Observation{Role: role, Text: text, New: true})
			}
			continue
		}
		o.worldMu.Lock()
		if text != "" {
			o.world[i] = worldState{text: text, at: now}
		}
		st, ok := o.world[i]
		o.worldMu.Unlock()
		if ok {
			out = append(out, Observation{Role: RoleWorld, Text: st.text, New: text != "", Age: now.Sub(st.at)})
		}
	}
	return out
}

// Buffers returns a snapshot of the latest buffer from each sensor, formatted as text.
func (o *Orchestrator) Buffers() []string {
	snapshot := make([]string, len(o.sensors))

	for i, sensor := range o.sensors {
		snapshot[i] = sensor.FormattedLatestBuffer()
	}

	return snapshot
}
