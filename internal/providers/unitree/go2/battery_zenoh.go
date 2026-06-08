package go2

import (
	"encoding/binary"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	defaultBatteryTopic = "lowstate"

	// rt/lowstate is published at ~500Hz. Decoding and locking on every sample
	// would peg a CPU core for data that changes slowly, so we drop samples that
	// arrive sooner than this interval after the last one we processed.
	batteryThrottleInterval = 200 * time.Millisecond

	batteryDebugLogInterval = 15 * time.Second

	// Absolute byte offsets of the battery-related fields within the CDR-encoded
	// unitree_go/msg/LowState payload (offsets include the 4-byte CDR
	// encapsulation header, matching the bytes delivered over zenoh).
	//
	// Verified against the cyclonedds reference serializer.
	lowStateSocOffset    = 1043 // bms_state.soc     uint8
	lowStateNTC1Offset   = 1152 // temperature_ntc1  uint8
	lowStateNTC2Offset   = 1153 // temperature_ntc2  uint8
	lowStatePowerVOffset = 1156 // power_v           float32 LE
	lowStatePowerAOffset = 1160 // power_a           float32 LE
	lowStateMinLen       = lowStatePowerAOffset + 4
)

// BatteryState is a snapshot of the Go2 battery readings.
type BatteryState struct {
	Percentage  float64 `json:"percentage"`
	Voltage     float64 `json:"voltage"`
	Amperes     float64 `json:"amperes"`
	Temperature int     `json:"temperature"`
}

// BatteryZenohProvider subscribes to rt/lowstate and maintains the latest
// battery snapshot, throttling the high-frequency stream to limit CPU usage.
type BatteryZenohProvider struct {
	log     *zap.Logger
	topic   string
	session zenohsession.Session
	sub     zenohsession.Subscriber

	// lastProcessedNs is the unix-nano timestamp of the last sample we decoded.
	// Read/written with atomics so the throttle check stays lock-free on the hot
	// path where most samples are discarded.
	lastProcessedNs atomic.Int64

	mu          sync.RWMutex
	percentage  float64
	voltage     float64
	amperes     float64
	temperature int

	lastDebugLog time.Time
}

// NewBatteryZenohProvider opens a zenoh session and subscribes to topic.
func NewBatteryZenohProvider(topic string) *BatteryZenohProvider {
	if topic == "" {
		topic = defaultBatteryTopic
	}

	p := &BatteryZenohProvider{log: logger.Get(), topic: topic}

	sess, err := zenohsession.Open()
	if err != nil {
		p.log.Warn("go2 battery: zenoh unavailable, provider disabled", zap.Error(err))
		return p
	}
	p.session = sess

	sub, err := sess.DeclareSubscriber(topic, p.onSample)
	if err != nil {
		sess.Close()
		p.session = nil
		p.log.Warn("go2 battery: failed to declare subscriber", zap.Error(err))
		return p
	}
	p.sub = sub

	p.log.Info("go2 battery: provider initialized", zap.String("topic", topic))
	return p
}

// onSample decodes an incoming LowState sample and updates the battery state.
//
// rt/lowstate arrives at ~500Hz; we discard samples that arrive within
// batteryThrottleInterval of the last one we accepted before doing any work.
func (p *BatteryZenohProvider) onSample(data []byte) {
	now := time.Now().UnixNano()
	last := p.lastProcessedNs.Load()
	if now-last < int64(batteryThrottleInterval) {
		return
	}
	// Claim this interval's slot; if a concurrent sample beat us, drop this one.
	if !p.lastProcessedNs.CompareAndSwap(last, now) {
		return
	}

	if len(data) < lowStateMinLen {
		p.log.Warn("go2 battery: payload too short", zap.Int("len", len(data)))
		return
	}

	soc := float64(data[lowStateSocOffset])
	ntc1 := int(data[lowStateNTC1Offset])
	ntc2 := int(data[lowStateNTC2Offset])
	powerV := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[lowStatePowerVOffset:])))
	powerA := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[lowStatePowerAOffset:])))

	p.mu.Lock()
	p.percentage = round2(soc)
	p.voltage = round2(powerV)
	p.amperes = round2(powerA)
	p.temperature = (ntc1 + ntc2) / 2

	if t := time.Now(); t.Sub(p.lastDebugLog) >= batteryDebugLogInterval {
		p.lastDebugLog = t
		p.log.Debug("go2 battery",
			zap.Float64("percentage", p.percentage),
			zap.Float64("voltage", p.voltage),
			zap.Float64("amperes", p.amperes),
			zap.Int("temperature", p.temperature),
		)
	}
	p.mu.Unlock()
}

// State returns the latest battery snapshot.
func (p *BatteryZenohProvider) State() BatteryState {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return BatteryState{
		Percentage:  p.percentage,
		Voltage:     p.voltage,
		Amperes:     p.amperes,
		Temperature: p.temperature,
	}
}

// Stop releases the zenoh subscriber and session.
func (p *BatteryZenohProvider) Stop() {
	if p.sub != nil {
		p.sub.Drop()
		p.sub = nil
	}

	if p.session != nil {
		p.session.Close()
		p.session = nil
	}

	p.log.Info("go2 battery: provider stopped")
}

// round2 rounds v to two decimal places.
func round2(v float64) float64 {
	return math.Round(v*100) / 100
}
