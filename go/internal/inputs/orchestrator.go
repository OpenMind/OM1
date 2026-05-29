package inputs

import (
	"context"
	"sync"

	"go.uber.org/zap"
)

type Orchestrator struct {
	sensors []Sensor
	tickNow chan struct{}
	log     *zap.Logger
}

func NewOrchestrator(sensors []Sensor, log *zap.Logger) *Orchestrator {
	return &Orchestrator{
		sensors: sensors,
		tickNow: make(chan struct{}, 1),
		log:     log,
	}
}

func (o *Orchestrator) TickNow() <-chan struct{} {
	return o.tickNow
}

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

func (o *Orchestrator) Buffers() []string {
	snapshot := make([]string, len(o.sensors))

	for i, sensor := range o.sensors {
		snapshot[i] = sensor.FormattedLatestBuffer()
	}

	return snapshot
}
