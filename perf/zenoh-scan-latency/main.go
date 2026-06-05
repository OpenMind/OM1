// Command zenoh-scan-latency subscribes to a zenoh topic (default: a ROS2
// LaserScan "scan" topic bridged onto zenoh) and measures, for every message,
// the delay between when it was sent and when this subscriber received it.
//
// "Sent time" is read from the ROS2 message header.stamp embedded in the CDR
// payload (bytes 4..11: int32 sec, uint32 nanosec, little-endian) — the same
// reference the Python counterpart uses, so the two are directly comparable.
//
// Build (needs the zenoh-c lib the main app already vendors):
//
//	make build            # sets up zenoh-c + CGO flags, then:
//	go build -o build/zenoh-scan-latency ./perf/zenoh-scan-latency
//
// Run:
//
//	./build/zenoh-scan-latency -key '<robot_ns>/pi/scan' -duration 30s
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/zenoh"
)

func main() {
	var (
		key      = flag.String("key", "**/scan", "zenoh key expression to subscribe to (e.g. <ns>/pi/scan)")
		endpoint = flag.String("endpoint", "tcp/127.0.0.1:7447", "zenoh router endpoint (client mode)")
		duration = flag.Duration("duration", 30*time.Second, "how long to sample before printing the summary")
		verbose  = flag.Bool("v", false, "print per-message delay")
	)
	flag.Parse()

	// internal/zenoh logs via the shared logger; initialize it so it never nil-panics.
	logger.Set(logger.BuildLogger("info"))

	sess, err := zenoh.Open(*endpoint)
	if err != nil {
		fmt.Printf("failed to open zenoh session: %v\n", err)
		return
	}
	defer sess.Close()

	var (
		mu      sync.Mutex
		delays  []float64 // milliseconds
		skipped int       // payloads too short to carry a header.stamp
	)

	sub, err := sess.DeclareSubscriber(*key, func(payload []byte) {
		recv := time.Now()
		if len(payload) < 12 {
			mu.Lock()
			skipped++
			mu.Unlock()
			return
		}
		// CDR: [0:4] encapsulation header, then std_msgs/Header.stamp.
		sec := int32(binary.LittleEndian.Uint32(payload[4:8]))
		nsec := binary.LittleEndian.Uint32(payload[8:12])
		sent := time.Unix(int64(sec), int64(nsec))
		ms := float64(recv.Sub(sent).Nanoseconds()) / 1e6

		mu.Lock()
		delays = append(delays, ms)
		mu.Unlock()

		if *verbose {
			fmt.Printf("delay=%.3f ms (sent=%s)\n", ms, sent.Format(time.RFC3339Nano))
		}
	})
	if err != nil {
		fmt.Printf("failed to subscribe to %q: %v\n", *key, err)
		return
	}
	defer sub.Drop()

	fmt.Printf("[go] subscribed to %q via %s — sampling for %s ...\n", *key, *endpoint, *duration)
	time.Sleep(*duration)

	mu.Lock()
	defer mu.Unlock()
	printSummary("go", *key, delays, skipped, *duration)
}

func printSummary(lang, key string, delays []float64, skipped int, dur time.Duration) {
	fmt.Printf("\n=== %s zenoh scan latency summary ===\n", lang)
	fmt.Printf("key: %s | window: %s | messages: %d | skipped(short): %d\n", key, dur, len(delays), skipped)
	if len(delays) == 0 {
		fmt.Println("no messages received — check the key expression and that the bridge is publishing")
		return
	}
	sorted := append([]float64(nil), delays...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, d := range delays {
		sum += d
	}
	rate := float64(len(delays)) / dur.Seconds()
	fmt.Printf("delay ms — min %.3f | p50 %.3f | p95 %.3f | p99 %.3f | max %.3f | mean %.3f\n",
		sorted[0], pct(sorted, 50), pct(sorted, 95), pct(sorted, 99), sorted[len(sorted)-1], sum/float64(len(delays)))
	fmt.Printf("message rate: %.1f msg/s\n", rate)
}

// pct returns the p-th percentile (0..100) of a pre-sorted slice via nearest-rank.
func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return math.NaN()
	}
	rank := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}
