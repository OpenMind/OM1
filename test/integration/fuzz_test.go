//go:build integration

package integration

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/runtime"
)

func FuzzModeSwitching(f *testing.F) {
	f.Add([]byte{3, 1, 4, 1, 5, 9, 2, 6})
	f.Add([]byte{0, 0, 0, 0, 0, 0})
	f.Add([]byte{25, 25, 25})
	f.Add([]byte{1, 30, 1, 30, 1, 30})

	f.Fuzz(func(t *testing.T, schedule []byte) {
		if len(schedule) == 0 {
			return
		}
		runFuzzModeSwitch(t, schedule)
	})
}

const maxFlips = 40

func runFuzzModeSwitch(t *testing.T, schedule []byte) {
	abs, err := filepath.Abs(filepath.Join("testdata", "fuzz_cases", "fuzz_mode_switch.json5"))
	require.NoError(t, err)
	cfg, err := config.Load(abs)
	require.NoError(t, err)

	session, s := newSession()
	injectActionMeta(cfg, session)

	drainModeContext()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rt := runtime.New(cfg, zap.NewNop(), runtime.Options{})
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		_ = rt.Run(ctx)
	}()

	var mu sync.Mutex
	target := "alpha"

	stopPub := make(chan struct{})
	var pubWG sync.WaitGroup
	pubWG.Add(1)
	go func() {
		defer pubWG.Done()
		tick := time.NewTicker(5 * time.Millisecond)
		defer tick.Stop()

		for {
			select {
			case <-stopPub:
				return
			case <-tick.C:
				mu.Lock()
				d := target
				mu.Unlock()
				providers.ModeContext().Publish(map[string]any{"target": d})
			}
		}
	}()

	flips := schedule
	if len(flips) > maxFlips {
		flips = flips[:maxFlips]
	}

	for _, b := range flips {
		time.Sleep(time.Duration(b%25) * time.Millisecond)
		mu.Lock()

		if target == "alpha" {
			target = "beta"
		} else {
			target = "alpha"
		}

		mu.Unlock()
	}

	time.Sleep(40 * time.Millisecond)

	close(stopPub)
	pubWG.Wait()

	cancel()
	select {
	case <-runDone:
	case <-time.After(20 * time.Second):
		t.Fatal("runtime did not shut down within 20s — possible deadlock under rapid transitions")
	}

	for _, c := range s.snapshot() {
		require.Containsf(t, []string{"alpha_act", "beta_act"}, c.Label,
			"recorded an action %q that belongs to no active mode", c.Label)
	}
}

func drainModeContext() {
	ch := providers.ModeContext().Updates()
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}
