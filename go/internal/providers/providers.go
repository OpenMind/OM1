package providers

import (
	"sync"
	"time"
)

type IOProvider struct {
	mu            sync.Mutex
	lastTickStart time.Time
	totalTicks    int64
}

var ioOnce sync.Once
var ioInstance *IOProvider

func IO() *IOProvider {
	ioOnce.Do(func() { ioInstance = &IOProvider{} })
	return ioInstance
}

func (p *IOProvider) RecordTick(start time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastTickStart = start
	p.totalTicks++
}

func (p *IOProvider) TotalTicks() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.totalTicks
}
