package g1

import (
	"context"
	"encoding/binary"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"

	bg "github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/hooks"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

const (
	// defaultTopic is the zenoh key the ros2 bridge republishes the hand
	// controller on. The DDS topic is rt/wirelesscontroller; the bridge keys by
	// the ROS name, so the "rt/" prefix is dropped, matching the other bridged
	// Unitree topics in this repo ("api/sport/request", "odom").
	defaultTopic = "wirelesscontroller"

	// debounce is the minimum gap between two accepted presses. The controller
	// streams at ~50 Hz and a press spans many frames, so rising-edge detection
	// alone can still double-fire if the contacts bounce.
	debounce = 500 * time.Millisecond

	// wirelessControllerLen is the CDR size of a WirelessController_ payload: a
	// 4-byte encapsulation header, four float32 stick axes, then a uint16 keymask.
	wirelessControllerLen = 4 + 4*4 + 2
)

// buttons is the Unitree wireless controller key bitmask, matching the
// xKeySwitchUnion layout used by the Unitree SDK.
type buttons uint16

var buttonBits = map[string]buttons{
	"r1": 1 << 0, "l1": 1 << 1, "start": 1 << 2, "select": 1 << 3,
	"r2": 1 << 4, "l2": 1 << 5, "f1": 1 << 6, "f2": 1 << 7,
	"a": 1 << 8, "b": 1 << 9, "x": 1 << 10, "y": 1 << 11,
	"up": 1 << 12, "right": 1 << 13, "down": 1 << 14, "left": 1 << 15,
}

// defaultButtons is A on its own.
var defaultButtons = []string{"A"}

func init() {
	bg.Register("UnitreeWirelessController", NewWirelessController)
}

// WirelessController starts a greeting when a hand-controller button combo is
// pressed, so an operator can open a conversation without waiting for the
// automatic person-detection flow.
type WirelessController struct {
	log     *zap.Logger
	session zenohsession.Session
	sub     zenohsession.Subscriber

	combo       buttons
	greetingCfg map[string]any

	ctx    context.Context
	cancel context.CancelFunc

	mu        sync.Mutex
	held      bool
	lastPress time.Time

	// inFlight keeps a slow greeting (face snapshot + LLM round trip) from
	// stacking up when the button is pressed repeatedly.
	inFlight atomic.Bool
}

// NewWirelessController subscribes to the controller topic. An unreachable
// zenoh session is logged and leaves the task inert rather than failing mode
// startup, matching the other zenoh-backed backgrounds. An unknown button name
// is a config error and fails loudly, since a silently dead button is worse.
func NewWirelessController(configMap map[string]any) (bg.Background, error) {
	log := logger.Get().Named("UnitreeWirelessController")

	names := stringSlice(configMap["buttons"])
	if len(names) == 0 {
		names = defaultButtons
	}
	combo, err := parseCombo(names)
	if err != nil {
		return nil, err
	}

	topic := defaultTopic
	if t, ok := configMap["topic"].(string); ok && strings.TrimSpace(t) != "" {
		topic = strings.TrimSpace(t)
	}

	greetingCfg, _ := configMap["greeting"].(map[string]any)

	ctx, cancel := context.WithCancel(context.Background())
	w := &WirelessController{
		log:         log,
		combo:       combo,
		greetingCfg: greetingCfg,
		ctx:         ctx,
		cancel:      cancel,
	}

	sess, err := zenohsession.Open()
	if err != nil {
		log.Error("zenoh unavailable, controller button disabled", zap.Error(err))
		return w, nil
	}
	w.session = sess

	sub, err := sess.DeclareSubscriber(topic, w.onFrame)
	if err != nil {
		log.Error("failed to declare subscriber", zap.Error(err), zap.String("topic", topic))
	} else {
		w.sub = sub
	}

	log.Info("background task initialized",
		zap.String("topic", topic), zap.Uint16("buttons", uint16(combo)))
	return w, nil
}

// onFrame decodes one controller sample and starts a greeting on the rising
// edge of the configured combo.
func (w *WirelessController) onFrame(data []byte) {
	keys, err := decodeKeys(data)
	if err != nil {
		w.log.Debug("failed to decode controller frame", zap.Error(err))
		return
	}

	if !w.pressed(keys, time.Now()) {
		return
	}

	w.log.Info("controller button pressed, starting greeting")
	w.startGreeting()
}

// pressed reports the debounced rising edge of the combo: it fires on the first
// frame the keymask matches, and stays quiet until it changes and matches again.
//
// The match is exact rather than a superset test, so that a single-button combo
// does not also fire as part of Unitree's own bindings — bare A must not trigger
// on L1+A (emergency damp) or SELECT+A (handshake).
func (w *WirelessController) pressed(keys buttons, now time.Time) bool {
	w.mu.Lock()
	defer w.mu.Unlock()

	matched := w.combo != 0 && keys == w.combo
	rising := matched && !w.held
	w.held = matched

	if !rising || now.Sub(w.lastPress) < debounce {
		return false
	}
	w.lastPress = now

	return true
}

// startGreeting runs off the subscriber callback, since generating a greeting
// involves a face snapshot and an LLM round trip and the zenoh callback must not
// block. Overlapping presses are dropped rather than queued.
func (w *WirelessController) startGreeting() {
	if !w.inFlight.CompareAndSwap(false, true) {
		w.log.Info("greeting already in progress, ignoring press")
		return
	}

	go func() {
		defer w.inFlight.Store(false)

		// Cut off anything already being said so the greeting is not queued
		// behind it, and lift any mute.
		tts.BargeIn()

		// Reopen the conversation so the turn counter restarts and the state
		// machine stops trying to wind the previous exchange down.
		providers.Greeting().ResetState(providers.StateEngaging)

		// Reuse the registered hook so a press produces the same greeting an
		// automatic mode entry would, memory recall included.
		if err := hooks.Invoke(w.ctx, "greeting_hook", "greeting_start_hook", w.greetingCfg, nil); err != nil {
			w.log.Error("greeting failed", zap.Error(err))
		}
	}()
}

// Run blocks until shutdown; work is driven by the subscriber callback.
func (w *WirelessController) Run(ctx context.Context) {
	<-ctx.Done()
}

// Stop releases the zenoh subscriber and session and cancels an in-flight greeting.
func (w *WirelessController) Stop() {
	w.log.Info("stopping background task")
	if w.sub != nil {
		w.sub.Drop()
		w.sub = nil
	}
	if w.session != nil {
		w.session.Close()
		w.session = nil
	}
	w.cancel()
}

// parseCombo converts config button names into a bitmask, case-insensitively.
func parseCombo(names []string) (buttons, error) {
	var combo buttons
	for _, raw := range names {
		name := strings.ToLower(strings.TrimSpace(raw))
		if name == "" {
			continue
		}
		bit, ok := buttonBits[name]
		if !ok {
			return 0, fmt.Errorf("unknown controller button %q", raw)
		}
		combo |= bit
	}
	return combo, nil
}

// decodeKeys extracts the keymask from a CDR little-endian
// unitree_go::msg::dds_::WirelessController_ payload, whose layout is a 4-byte
// encapsulation header, lx/ly/rx/ry as float32, then keys as uint16.
func decodeKeys(data []byte) (buttons, error) {
	if len(data) < wirelessControllerLen {
		return 0, fmt.Errorf("wireless controller: payload too short (%d bytes)", len(data))
	}
	return buttons(binary.LittleEndian.Uint16(data[20:])), nil
}

// stringSlice coerces a decoded JSON5 array into a []string.
func stringSlice(v any) []string {
	vals, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(vals))
	for _, item := range vals {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
