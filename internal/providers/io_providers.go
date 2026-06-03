package providers

import (
	"sync"
	"time"
)

// Input represents an input value along with its timestamp and the tick counter value when it was recorded.
type Input struct {
	Input     string
	Timestamp time.Time
	Tick      int
}

// IOProvider is a singleton that manages input storage and tick counting for the runtime.
type IOProvider struct {
	mu            sync.Mutex
	lastTickStart time.Time
	totalTicks    int64

	inputs      map[string]Input
	dynamicVars map[string]string
	tickCounter int
}

var ioOnce sync.Once
var ioInstance *IOProvider

func IO() *IOProvider {
	ioOnce.Do(func() {
		ioInstance = &IOProvider{
			inputs:      make(map[string]Input),
			dynamicVars: make(map[string]string),
		}
	})
	return ioInstance
}

// SetDynamicVar stores a dynamic variable under key.
func (p *IOProvider) SetDynamicVar(key, value string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.dynamicVars[key] = value
}

// GetDynamicVar returns the dynamic variable for key. The bool is false if absent.
func (p *IOProvider) GetDynamicVar(key string) (string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	v, ok := p.dynamicVars[key]
	return v, ok
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

// AddInput stores an input under key with the given value, recording the current
// tick counter.
func (p *IOProvider) AddInput(key, value string, timestamp time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if timestamp.IsZero() {
		timestamp = time.Now()
	}
	p.inputs[key] = Input{Input: value, Timestamp: timestamp, Tick: p.tickCounter}
}

// GetInput returns a copy of the input stored under key, or nil if absent.
func (p *IOProvider) GetInput(key string) *Input {
	p.mu.Lock()
	defer p.mu.Unlock()
	in, ok := p.inputs[key]
	if !ok {
		return nil
	}
	return &in
}

// RemoveInput removes the input stored under key, if any.
func (p *IOProvider) RemoveInput(key string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.inputs, key)
}

// AddInputTimestamp updates the timestamp of an existing input, preserving its
// value and tick. It is a no-op if the key is absent.
func (p *IOProvider) AddInputTimestamp(key string, timestamp time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if in, ok := p.inputs[key]; ok {
		in.Timestamp = timestamp
		p.inputs[key] = in
	}
}

// GetInputTimestamp returns the timestamp for key. The bool is false if the key
// is absent.
func (p *IOProvider) GetInputTimestamp(key string) (time.Time, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	in, ok := p.inputs[key]
	if !ok {
		return time.Time{}, false
	}
	return in.Timestamp, true
}

// TickCounter returns the current tick counter value.
func (p *IOProvider) TickCounter() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tickCounter
}

// IncrementTick increments the tick counter and returns the new value.
func (p *IOProvider) IncrementTick() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tickCounter++
	return p.tickCounter
}

// ResetTickCounter resets the tick counter to zero.
func (p *IOProvider) ResetTickCounter() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.tickCounter = 0
}
