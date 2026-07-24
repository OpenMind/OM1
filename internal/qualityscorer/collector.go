package qualityscorer

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// recentWindow matches dataAnalysis's live_quality_scorer.py RECENT_WINDOW.
const recentWindow = time.Hour

// coherenceScore mirrors live_quality_scorer.py's COHERENCE_SCORE.
var coherenceScore = map[string]float64{
	"incoherent": 0.0,
	"marginal":   0.5,
	"coherent":   1.0,
}

// scoreEvent is one scored turn, kept in memory only long enough to compute
// the trailing-window metrics below.
type scoreEvent struct {
	at             time.Time
	language       string // "" if undetected
	classification string // "" only if extraction somehow produced no label (shouldn't happen)
	coherence      string // "" if not scored (no robot response that turn)
}

// liveCollector is the Go equivalent of live_quality_scorer.py's
// LiveQualityCollector: registered once on prometheus.DefaultRegisterer (the
// same registry internal/metrics.StartServer already serves at :9090/metrics),
// its Collect() recomputes every metric fresh on each scrape from in-memory
// state -- no disk re-read, since (unlike the Python original, which had to
// survive independent restarts of just the scorer process) a restart of this
// process means a fresh session for everything, tracer included.
type liveCollector struct {
	mu sync.Mutex

	allLanguages       map[string]int
	allClassifications map[string]int
	allCoherence       map[string]int
	turnsScored        int
	activeScore        *float64

	recent []scoreEvent // time-ordered; pruned to recentWindow on each append
}

func newLiveCollector() *liveCollector {
	return &liveCollector{
		allLanguages:       map[string]int{},
		allClassifications: map[string]int{},
		allCoherence:       map[string]int{},
	}
}

// record folds one scored turn into the running totals. at should be the
// time scoring completed (matches the Python original's scored_at, not the
// original trace timestamp).
func (c *liveCollector) record(ev scoreEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if ev.language != "" {
		c.allLanguages[ev.language]++
	}
	if ev.classification != "" {
		c.allClassifications[ev.classification]++
	}
	if ev.coherence != "" {
		c.allCoherence[ev.coherence]++
		c.turnsScored++
		score := coherenceScore[ev.coherence]
		c.activeScore = &score
	}

	c.recent = append(c.recent, ev)
	cutoff := ev.at.Add(-recentWindow)
	i := 0
	for i < len(c.recent) && c.recent[i].at.Before(cutoff) {
		i++
	}
	c.recent = c.recent[i:]
}

// Describe intentionally sends nothing -- this is an "unchecked" collector
// (dynamic label values discovered only at Collect time), the same trade-off
// Python's ad hoc GaugeMetricFamily-per-scrape design made.
func (c *liveCollector) Describe(ch chan<- *prometheus.Desc) {}

func (c *liveCollector) Collect(ch chan<- prometheus.Metric) {
	c.mu.Lock()
	allLanguages := cloneCounts(c.allLanguages)
	allClassifications := cloneCounts(c.allClassifications)
	allCoherence := cloneCounts(c.allCoherence)
	turnsScored := c.turnsScored
	activeScore := c.activeScore
	recent := append([]scoreEvent(nil), c.recent...)
	c.mu.Unlock()

	cutoff := time.Now().Add(-recentWindow)
	recentLanguages := map[string]int{}
	recentClassifications := map[string]int{}
	recentCoherence := map[string]int{}
	for _, ev := range recent {
		if ev.at.Before(cutoff) {
			continue
		}
		if ev.language != "" {
			recentLanguages[ev.language]++
		}
		if ev.classification != "" {
			recentClassifications[ev.classification]++
		}
		if ev.coherence != "" {
			recentCoherence[ev.coherence]++
		}
	}

	emitCounts(ch, "om1_quality_live_language_count",
		"All-time count of scored turns per detected spoken language", "language", allLanguages)
	emitCounts(ch, "om1_quality_live_language_count_last_1h",
		"Count of scored turns per detected spoken language in the trailing hour", "language", recentLanguages)

	emitCounts(ch, "om1_quality_live_input_classification_count",
		"All-time count of user inputs per classification label (positive/marginal/negative/not_addressed)", "label", allClassifications)
	emitCounts(ch, "om1_quality_live_input_classification_count_last_1h",
		"Count of user inputs per classification label in the trailing hour", "label", recentClassifications)

	emitCounts(ch, "om1_quality_live_coherence_count",
		"All-time count of prompt/response pairs per coherence label (coherent/marginal/incoherent)", "coherence", allCoherence)
	emitCounts(ch, "om1_quality_live_coherence_count_last_1h",
		"Count of prompt/response pairs per coherence label in the trailing hour", "coherence", recentCoherence)

	ch <- prometheus.MustNewConstMetric(
		prometheus.NewDesc("om1_quality_live_turns_scored", "All-time count of turns that produced a coherence score", nil, nil),
		prometheus.CounterValue, float64(turnsScored),
	)

	if activeScore != nil {
		ch <- prometheus.MustNewConstMetric(
			prometheus.NewDesc("om1_quality_live_active_score", "Most recent turn's coherence score: coherent=1, marginal=0.5, incoherent=0", nil, nil),
			prometheus.GaugeValue, *activeScore,
		)
	}
}

func cloneCounts(m map[string]int) map[string]int {
	out := make(map[string]int, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func emitCounts(ch chan<- prometheus.Metric, name, help, labelName string, counts map[string]int) {
	desc := prometheus.NewDesc(name, help, []string{labelName}, nil)
	for label, count := range counts {
		ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(count), label)
	}
}
