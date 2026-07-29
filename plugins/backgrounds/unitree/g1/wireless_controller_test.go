package g1

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// frame builds a CDR WirelessController_ payload carrying the given keymask.
func frame(keys buttons) []byte {
	buf := make([]byte, wirelessControllerLen)
	buf[1] = 0x01 // little-endian CDR encapsulation
	binary.LittleEndian.PutUint16(buf[20:], uint16(keys))
	return buf
}

func testController(t *testing.T) *WirelessController {
	t.Helper()

	combo, err := parseCombo(defaultButtons)
	require.NoError(t, err)

	return &WirelessController{combo: combo}
}

// btnA is the A button's bit, the default trigger.
const btnA = buttons(1 << 8)

func TestButtonBitsMatchUnitreeSDK(t *testing.T) {
	// Confirmed against the hardware: SELECT reads 8 and B reads 512, so the
	// documented xKeySwitchUnion bit order holds.
	require.Equal(t, buttons(8), buttonBits["select"])
	require.Equal(t, buttons(512), buttonBits["b"])
	require.Equal(t, buttons(256), buttonBits["a"])

	combo, err := parseCombo(defaultButtons)
	require.NoError(t, err)
	require.Equal(t, btnA, combo)
}

func TestParseCombo(t *testing.T) {
	combo, err := parseCombo([]string{" A "})
	require.NoError(t, err, "names are case- and space-insensitive")
	require.Equal(t, btnA, combo)

	_, err = parseCombo([]string{"SELECT", "Z"})
	require.ErrorContains(t, err, "Z", "a typo fails loudly rather than disabling the button")
}

func TestDecodeKeys(t *testing.T) {
	keys, err := decodeKeys(frame(btnA))
	require.NoError(t, err)
	require.Equal(t, btnA, keys)

	// A bridge may append padding; trailing bytes must not break decoding.
	keys, err = decodeKeys(append(frame(btnA), 0, 0))
	require.NoError(t, err)
	require.Equal(t, btnA, keys)

	_, err = decodeKeys(make([]byte, 21))
	require.ErrorContains(t, err, "too short")
}

func TestPressFiresOnRisingEdge(t *testing.T) {
	w := testController(t)
	now := time.Unix(0, 0)

	require.False(t, w.pressed(0, now), "idle")
	require.True(t, w.pressed(btnA, now), "A pressed")
}

func TestHeldButtonFiresOnlyOnce(t *testing.T) {
	w := testController(t)
	now := time.Unix(0, 0)

	// The controller streams at ~50 Hz, so a one-second hold is ~50 frames.
	fired := 0
	for range 50 {
		if w.pressed(btnA, now) {
			fired++
		}
		now = now.Add(20 * time.Millisecond)
	}

	require.Equal(t, 1, fired, "holding must not re-fire every frame")
}

func TestReleaseThenPressFiresAgain(t *testing.T) {
	w := testController(t)
	now := time.Unix(0, 0)

	require.True(t, w.pressed(btnA, now))
	require.False(t, w.pressed(0, now))
	require.True(t, w.pressed(btnA, now.Add(debounce)))
}

func TestDebounceSuppressesBouncedPress(t *testing.T) {
	w := testController(t)
	now := time.Unix(0, 0)

	// A bouncing contact looks like press/release/press within a few ms.
	require.True(t, w.pressed(btnA, now))
	require.False(t, w.pressed(0, now.Add(5*time.Millisecond)))
	require.False(t, w.pressed(btnA, now.Add(10*time.Millisecond)),
		"the second edge falls inside the debounce window")
}

func TestUnitreeOwnBindingsAreIgnored(t *testing.T) {
	w := testController(t)
	now := time.Unix(0, 0)

	// Unitree's own bindings must pass through untouched. L1+A and SELECT+A
	// both contain A, so an exact match is what keeps EMERGENCY DAMP and
	// handshake from also starting a conversation.
	for _, keys := range []buttons{
		1<<1 | btnA,  // L1+A, emergency damp
		1<<3 | btnA,  // SELECT+A, handshake
		1<<1 | 1<<12, // L1+UP, lock stand
		1<<4 | 1<<10, // R2+X, start motion control
		1 << 2,       // start
	} {
		require.False(t, w.pressed(keys, now), "keymask %d must not fire", keys)
		w.pressed(0, now)
	}
}

func TestStickMotionDoesNotFire(t *testing.T) {
	w := testController(t)

	// Walking the robot around must never be mistaken for a button press: the
	// stick axes occupy the bytes before the keymask.
	buf := make([]byte, wirelessControllerLen)
	buf[1] = 0x01
	for i := 4; i < 20; i++ {
		buf[i] = 0xFF
	}

	keys, err := decodeKeys(buf)
	require.NoError(t, err)
	require.Zero(t, keys)
	require.False(t, w.pressed(keys, time.Unix(0, 0)))
}

func TestConfigDefaults(t *testing.T) {
	// The config entry carries only what differs from the defaults, so an
	// absent or empty config map must still yield a working button.
	for _, configMap := range []map[string]any{nil, {}} {
		names := stringSlice(configMap["buttons"])
		if len(names) == 0 {
			names = defaultButtons
		}
		combo, err := parseCombo(names)
		require.NoError(t, err)
		require.Equal(t, btnA, combo)

		_, ok := configMap["topic"].(string)
		require.False(t, ok, "topic falls back to the bridged zenoh key")
	}
}

func TestStringSlice(t *testing.T) {
	require.Equal(t, []string{"SELECT", "B"}, stringSlice([]any{"SELECT", "B"}))
	require.Equal(t, []string{"a"}, stringSlice([]any{"a", 42}))
	require.Nil(t, stringSlice(nil))
}
