// Package traceexport broadcasts live LLM trace records as Prometheus
// series on GET /traces/metrics, for a co-located telemetry sidecar to poll
// and archive to its own storage -- see internal/metrics's traceRegistry
// doc comment for why this stays off the default /metrics registry.
package traceexport

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/tracer/tracetype"
)

// maxBuffered bounds how many recent trace records stay exposed on
// /traces/metrics at once. Each record is its own Prometheus series until
// evicted, so this caps memory even if no sidecar ever polls. At OM1's
// typical trace rate (roughly one per LLM turn) this is many hours of
// headroom -- a sidecar just needs to poll more often than that.
const maxBuffered = 200

// labelValues is one record's Prometheus label values, in the exact order
// metrics.TraceInfo declares them -- kept so eviction can call
// DeleteLabelValues with the same tuple used to create the series.
type labelValues [5]string

// Start begins exporting trace records from records as Prometheus series,
// evicting the oldest once maxBuffered is exceeded. Runs until records is
// closed or ctx is cancelled.
func Start(ctx context.Context, records <-chan tracetype.TraceRecord, log *zap.Logger) {
	log.Info("traceexport: started, exporting live trace records on /traces/metrics")

	go func() {
		var window []labelValues
		var nextSeq int64

		for {
			select {
			case <-ctx.Done():
				return
			case rec, ok := <-records:
				if !ok {
					return
				}

				outputJSON, err := json.Marshal(rec.LLMOutput)
				if err != nil {
					log.Warn("traceexport: failed to marshal llm_output, skipping record", zap.Error(err))
					continue
				}

				// Prometheus label values must be valid UTF-8 -- WithLabelValues
				// panics otherwise (crashing this whole process, not just this
				// goroutine). Truncated multi-byte characters upstream (e.g. a
				// memory snippet cut at a byte boundary) can produce invalid
				// UTF-8 in a prompt, so scrub both text fields defensively
				// rather than trust every upstream source to hand back clean text.
				lv := labelValues{
					strconv.FormatInt(nextSeq, 10),
					rec.Timestamp,
					strconv.Itoa(rec.Generation),
					strings.ToValidUTF8(rec.LLMInput, "�"),
					strings.ToValidUTF8(string(outputJSON), "�"),
				}
				nextSeq++

				metrics.TraceInfo.WithLabelValues(lv[0], lv[1], lv[2], lv[3], lv[4]).Set(1)
				window = append(window, lv)

				for len(window) > maxBuffered {
					evict := window[0]
					window = window[1:]
					metrics.TraceInfo.DeleteLabelValues(evict[0], evict[1], evict[2], evict[3], evict[4])
				}
			}
		}
	}()
}
