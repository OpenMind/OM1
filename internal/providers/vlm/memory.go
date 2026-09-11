package vlm

import (
	"sync"
	"time"
)

// Sampling strategies for deciding which retained frames are re-sent with the
// current one. Both perform comparably in practice; uniform gives an even view
// of the window, recency weights the last few seconds more heavily.
const (
	SamplingUniform = "uniform"
	SamplingRecency = "recency"
)

const (
	defaultMemoryFrames      = 3
	defaultMemoryTexts       = 3
	defaultMemoryIntervalSec = 2.0
	defaultMemoryCapacity    = 12
)

// MemoryOptions is the JSON-configurable form of the memory harness. Pointer
// fields distinguish "absent" (use the default) from an explicit 0, so
// memory_frames: 0 turns image memory off and restores single-frame requests.
type MemoryOptions struct {
	// Frames is the number of past frames re-sent alongside the current one.
	Frames *int `json:"memory_frames"`
	// Texts is the number of past descriptions included as a text cache.
	Texts *int `json:"memory_texts"`
	// IntervalSec is the minimum spacing between retained steps, which sets how
	// far back the memory window reaches.
	IntervalSec *float64 `json:"memory_interval_sec"`
	// Capacity is the number of steps retained before the oldest is evicted.
	Capacity *int `json:"memory_capacity"`
	// Sampling is "uniform" or "recency".
	Sampling string `json:"memory_sampling"`
}

// Config resolves the options into a MemoryConfig, applying defaults.
func (o MemoryOptions) Config() MemoryConfig {
	cfg := MemoryConfig{
		MaxFrames:   defaultMemoryFrames,
		MaxTexts:    defaultMemoryTexts,
		MinInterval: time.Duration(defaultMemoryIntervalSec * float64(time.Second)),
		Capacity:    defaultMemoryCapacity,
		Sampling:    o.Sampling,
	}
	if o.Frames != nil {
		cfg.MaxFrames = *o.Frames
	}
	if o.Texts != nil {
		cfg.MaxTexts = *o.Texts
	}
	if o.IntervalSec != nil {
		cfg.MinInterval = time.Duration(*o.IntervalSec * float64(time.Second))
	}
	if o.Capacity != nil {
		cfg.Capacity = *o.Capacity
	}
	return cfg
}

// MemoryConfig configures the memory harness.
type MemoryConfig struct {
	MaxFrames   int
	MaxTexts    int
	MinInterval time.Duration
	Capacity    int
	Sampling    string
}

// HistoryStep is one recalled step: a past observation, the description the
// model produced for it, and how long before the current frame it happened.
type HistoryStep struct {
	Age         time.Duration
	JPEGBase64  string
	Description string
}

// History is the context the harness re-injects into a request: a sampled set
// of past frames with their descriptions, plus a text-only cache of the most
// recent descriptions whose frames were not sampled.
type History struct {
	Frames []HistoryStep
	Texts  []HistoryStep
}

// Empty reports whether there is nothing to re-inject.
func (h History) Empty() bool {
	return len(h.Frames) == 0 && len(h.Texts) == 0
}

// memoryStep is one retained step of the trajectory.
type memoryStep struct {
	ts    time.Time
	jpeg  string
	descr string
}

// Memory is a bounded, thread-safe record of past observation/description
// pairs. Steps are retained no closer together than MinInterval, so a small
// buffer still spans a useful stretch of time.
type Memory struct {
	cfg MemoryConfig

	mu    sync.Mutex
	steps []memoryStep // oldest first
}

// NewMemory constructs a Memory, clamping the config to sane bounds.
func NewMemory(cfg MemoryConfig) *Memory {
	if cfg.MaxFrames < 0 {
		cfg.MaxFrames = 0
	}
	if cfg.MaxTexts < 0 {
		cfg.MaxTexts = 0
	}
	if cfg.MinInterval < 0 {
		cfg.MinInterval = 0
	}
	if cfg.Sampling != SamplingRecency {
		cfg.Sampling = SamplingUniform
	}
	return &Memory{cfg: cfg}
}

// Enabled reports whether the harness contributes anything to a request.
func (m *Memory) Enabled() bool {
	return m != nil && m.cfg.Capacity > 0 && (m.cfg.MaxFrames > 0 || m.cfg.MaxTexts > 0)
}

// Add records an observation and the description produced for it. Steps closer
// than MinInterval to the newest retained step are dropped, and the oldest step
// is evicted once capacity is reached.
func (m *Memory) Add(ts time.Time, jpegBase64, description string) {
	if !m.Enabled() || description == "" {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if n := len(m.steps); n > 0 {
		last := m.steps[n-1].ts
		if ts.Sub(last) < m.cfg.MinInterval {
			return
		}
	}

	m.steps = append(m.steps, memoryStep{ts: ts, jpeg: jpegBase64, descr: description})
	if len(m.steps) > m.cfg.Capacity {
		m.steps = m.steps[len(m.steps)-m.cfg.Capacity:]
	}
}

// History samples the retained steps into the context for the next request.
// Ages are reported relative to now, which callers set to the timestamp of the
// frame they are about to send.
func (m *Memory) History(now time.Time) History {
	if !m.Enabled() {
		return History{}
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	n := len(m.steps)
	if n == 0 {
		return History{}
	}

	var hist History
	sampled := make(map[int]bool, m.cfg.MaxFrames)
	for _, i := range sampleIndices(n, m.cfg.MaxFrames, m.cfg.Sampling) {
		if m.steps[i].jpeg == "" {
			continue
		}
		sampled[i] = true
		hist.Frames = append(hist.Frames, m.step(i, now))
	}

	start := n - m.cfg.MaxTexts
	if start < 0 {
		start = 0
	}
	for i := start; i < n; i++ {
		if sampled[i] {
			continue
		}
		step := m.step(i, now)
		step.JPEGBase64 = ""
		hist.Texts = append(hist.Texts, step)
	}

	return hist
}

// step converts a retained step into a HistoryStep. Caller holds the lock.
func (m *Memory) step(i int, now time.Time) HistoryStep {
	s := m.steps[i]
	age := now.Sub(s.ts)
	if age < 0 {
		age = 0
	}
	return HistoryStep{Age: age, JPEGBase64: s.jpeg, Description: s.descr}
}

// sampleIndices picks up to k indices from [0, n) in ascending order. Both the
// oldest and the newest retained step are always included when k allows, so the
// request keeps the earliest state of the window as an anchor.
func sampleIndices(n, k int, sampling string) []int {
	if n <= 0 || k <= 0 {
		return nil
	}
	if k >= n {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}
	if k == 1 {
		return []int{n - 1}
	}

	picked := make(map[int]bool, k)
	if sampling == SamplingRecency {
		picked[n-1] = true
		for off := 1; len(picked) < k-1 && off <= n-1; off *= 2 {
			picked[n-1-off] = true
		}
		picked[0] = true
	} else {
		for j := 0; j < k; j++ {
			// Evenly spaced across the window, endpoints included.
			picked[(j*(n-1)*2+(k-1))/(2*(k-1))] = true
		}
	}

	out := make([]int, 0, len(picked))
	for i := 0; i < n; i++ {
		if picked[i] {
			out = append(out, i)
		}
	}
	return out
}
