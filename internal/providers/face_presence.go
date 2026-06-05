// Package providers (face presence) — drop into internal/providers/.
//
// FacePresenceProvider polls the face HTTP service's /who endpoint and
// renders the snapshot into LLM-readable text. In passive mode (LLM
// doesn't tick on face change), the text the LLM sees here on its next
// user-driven tick is its only window into the room. So the format
// encodes everything the LLM needs to choose its greeting style.
//
// GREETING-STYLE LADDER
// ---------------------
//
// For NAMED faces, the LLM picks one of three styles based on how long
// ago this identity was last seen:
//
//   < 1 day      → "Welcome back, <Name>!" (recent, conversational)
//   1 ≤ d < 7    → "Hi <Name>, I've seen you N days ago." (recent-ish)
//   ≥ 7 days     → "Hi <Name>, last time we met was on <DATE>."
//
// For ANON faces (auto-enrolled, no name yet), two states:
//
//   < 3 min      → "Hi! What's your name?" (newcomer, just discovered)
//   ≥ 3 min      → "We've met before — what's your name?"
//
// UNKNOWN faces (no UUID assigned yet, still in the 1-3s window between
// detection and auto-enroll) are HIDDEN from the text — they're a
// transitional state the LLM can't act on (no identity to name).
//
// STICKY SESSION SEMANTICS
// ------------------------
// The "last seen" gap is snapshotted at the start of each track session,
// not the current gallery value (which is touched every frame the face
// is visible). Same person staying in the room for an hour: gap stays
// constant. Same person leaving and returning: gap reflects the
// PREVIOUS visit, not "0 seconds ago".
package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// FacePresenceConfig configures the provider.
type FacePresenceConfig struct {
	BaseURL   string
	RecentSec float64
	Timeout   time.Duration

	// Thresholds (seconds). 0 → use defaults.
	AnonNewcomerThrSec     float64 // default 180 (3 min)
	NamedRecentThrSec      float64 // default 86_400 (1 day)
	NamedLongAbsenceThrSec float64 // default 604_800 (7 days)
}

// FacePresenceProvider fetches /who snapshots and renders them as text.
type FacePresenceProvider struct {
	cfg    FacePresenceConfig
	client *http.Client
}

// NewFacePresenceProvider constructs a provider.
func NewFacePresenceProvider(cfg FacePresenceConfig) *FacePresenceProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://127.0.0.1:6793"
	}
	if cfg.RecentSec <= 0 {
		cfg.RecentSec = 1.0
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 2 * time.Second
	}
	if cfg.AnonNewcomerThrSec <= 0 {
		cfg.AnonNewcomerThrSec = 180.0
	}
	if cfg.NamedRecentThrSec <= 0 {
		cfg.NamedRecentThrSec = 86_400.0
	}
	if cfg.NamedLongAbsenceThrSec <= 0 {
		cfg.NamedLongAbsenceThrSec = 604_800.0
	}
	return &FacePresenceProvider{
		cfg:    cfg,
		client: &http.Client{Timeout: cfg.Timeout},
	}
}

// FaceEntry is one face in a /who snapshot. Field names mirror the JSON
// returned by the Python API. Pointer types let us distinguish "field
// absent / null" from "field present with value 0".
type FaceEntry struct {
	Name           string   `json:"name"`              // "sean" / "anon_73d0a4" / "unknown"
	UUID           string   `json:"uuid"`              // full 32-hex; "" for unknown
	Tier           string   `json:"tier"`              // "confident" / "tentative" / "uncertain"
	Sim            float64  `json:"sim"`               // raw cosine sim
	TrackID        int      `json:"track_id"`
	Area           int      `json:"area,omitempty"`
	CreatedAgoSec  *float64 `json:"created_ago_sec"`   // UUID age in seconds; null for unknown
	LastSeenAgoSec *float64 `json:"last_seen_ago_sec"` // sticky session-start gap; null until first confident match
	LastSeenISO    *string  `json:"last_seen_iso"`     // ISO timestamp of previous sighting
}

// PresenceSnapshot is the parsed /who response (we only need a subset of fields).
//
// ClosestName / ClosestUUID / ClosestTier are derived fields populated
// by FetchSnapshot after parsing — they describe the most prominent
// (largest-area) face on screen. Kept here for backward compatibility
// with code that worked with the previous "single closest face" model
// (e.g. lifecycle hooks deciding how to greet the room on startup).
// New code should iterate ``Faces`` instead.
type PresenceSnapshot struct {
	OK       bool        `json:"ok"`
	Faces    []FaceEntry `json:"faces"`
	ServerTS float64     `json:"server_ts"`

	// Derived (populated by FetchSnapshot, not from JSON).
	ClosestName string  `json:"-"`
	ClosestUUID string  `json:"-"`
	ClosestTier string  `json:"-"`
	ClosestSim  float64 `json:"-"`
}

// providerForDefaults exposes the configured thresholds to the
// PresenceSnapshot.ToText method (which is a method on PresenceSnapshot, not the
// provider, for simple test ergonomics). face_presence.go calls
// SetDefaults at startup so these match the parsed runtime config.
var providerForDefaults = &FacePresenceProvider{
	cfg: FacePresenceConfig{
		AnonNewcomerThrSec:     180.0,
		NamedRecentThrSec:      86_400.0,
		NamedLongAbsenceThrSec: 604_800.0,
	},
}

// SetDefaults updates the package-level thresholds (call once at startup).
func SetDefaults(cfg FacePresenceConfig) {
	if cfg.AnonNewcomerThrSec > 0 {
		providerForDefaults.cfg.AnonNewcomerThrSec = cfg.AnonNewcomerThrSec
	}
	if cfg.NamedRecentThrSec > 0 {
		providerForDefaults.cfg.NamedRecentThrSec = cfg.NamedRecentThrSec
	}
	if cfg.NamedLongAbsenceThrSec > 0 {
		providerForDefaults.cfg.NamedLongAbsenceThrSec = cfg.NamedLongAbsenceThrSec
	}
}

// FetchSnapshot calls POST /who and returns the parsed result.
//
// Returns by VALUE rather than pointer for backward compatibility with
// hook code that takes ``PresenceSnapshot`` arguments. On error, the
// returned snapshot is the zero value (Faces=nil, OK=false), and the
// error is non-nil.
//
// Populates the derived ``Closest*`` fields from the largest-area face
// before returning, so hook code can call ``snap.ClosestName`` without
// iterating ``Faces``.
func (p *FacePresenceProvider) FetchSnapshot(ctx context.Context) (PresenceSnapshot, error) {
	body, _ := json.Marshal(map[string]any{"recent_sec": p.cfg.RecentSec})
	req, err := http.NewRequestWithContext(
		ctx, "POST", p.cfg.BaseURL+"/who", bytes.NewReader(body),
	)
	if err != nil {
		return PresenceSnapshot{}, fmt.Errorf("build /who request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return PresenceSnapshot{}, fmt.Errorf("/who request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return PresenceSnapshot{}, fmt.Errorf("/who body read: %w", err)
	}
	var snap PresenceSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return PresenceSnapshot{}, fmt.Errorf("/who decode: %w (body=%s)", err, string(data))
	}

	// Populate Closest* derived fields from the largest visible face
	// that has a UUID. Falls through to empty strings if no such face.
	//
	// For hook usage: ``ClosestName`` is meant as a person's NAME for
	// natural-language greetings ("Hello Sean"). Auto-enrolled "anon_xxx"
	// labels are not names — surfacing them to the LLM would produce
	// awkward output like "Hello anon_73d0a4". So when the closest face
	// is anonymous, we leave ``ClosestName`` empty (still populate
	// ``ClosestUUID``/``ClosestTier``/``ClosestSim`` for diagnostics).
	// The hook's "no one in particular" fallback then kicks in.
	var bestArea int
	for _, f := range snap.Faces {
		if f.UUID == "" {
			continue
		}
		if f.Area > bestArea {
			bestArea = f.Area
			snap.ClosestUUID = f.UUID
			snap.ClosestTier = f.Tier
			snap.ClosestSim = f.Sim
			if strings.HasPrefix(f.Name, "anon_") {
				snap.ClosestName = ""
			} else {
				snap.ClosestName = f.Name
			}
		}
	}
	return snap, nil
}

// ToText renders the snapshot into one LLM-readable line.
//
// Returns "" when no actionable faces (e.g. only "unknown" transients)
// so the caller can treat empty as "nothing new" and keep the prompt
// clean during idle periods.
func (s *PresenceSnapshot) ToText() string {
	if s == nil || len(s.Faces) == 0 {
		return ""
	}

	// Sort by area desc (largest face = most prominent / closest).
	faces := make([]FaceEntry, len(s.Faces))
	copy(faces, s.Faces)
	sort.SliceStable(faces, func(i, j int) bool {
		if faces[i].Area != faces[j].Area {
			return faces[i].Area > faces[j].Area
		}
		return faces[i].TrackID < faces[j].TrackID
	})

	// Bucket. Unknown faces are intentionally dropped (transient state
	// between detection and auto-enroll, no actionable UUID yet).
	var named, anons []FaceEntry
	for _, f := range faces {
		switch {
		case strings.HasPrefix(f.Name, "anon_"):
			anons = append(anons, f)
		case f.Name == "unknown" || f.Name == "":
			// drop
		default:
			named = append(named, f)
		}
	}

	if len(named) == 0 && len(anons) == 0 {
		return ""
	}

	total := len(named) + len(anons)
	descriptor := fmt.Sprintf("FacePresence: %d face", total)
	if total != 1 {
		descriptor += "s"
	}

	var parts []string
	for _, f := range named {
		parts = append(parts, formatNamedEntry(f))
	}
	for _, f := range anons {
		parts = append(parts, formatAnonEntry(f))
	}

	return fmt.Sprintf("%s — %s", descriptor, strings.Join(parts, ", "))
}

// formatNamedEntry — three-tier label based on how long since last sighting.
//
//   "sean (recognized)"                              — < 1 day
//   "sean (recognized, 2 days ago)"                  — 1-6 days
//   "sean (recognized, last seen 2026-03-05)"        — ≥ 7 days
func formatNamedEntry(f FaceEntry) string {
	cfg := providerForDefaults.cfg
	if f.LastSeenAgoSec == nil {
		return fmt.Sprintf("%s (recognized)", f.Name)
	}
	gap := *f.LastSeenAgoSec
	switch {
	case gap < cfg.NamedRecentThrSec:
		return fmt.Sprintf("%s (recognized)", f.Name)
	case gap < cfg.NamedLongAbsenceThrSec:
		days := int(gap / 86_400.0)
		if days <= 1 {
			return fmt.Sprintf("%s (recognized, 1 day ago)", f.Name)
		}
		return fmt.Sprintf("%s (recognized, %d days ago)", f.Name, days)
	default:
		dateStr := lastSeenDate(f.LastSeenISO)
		if dateStr == "" {
			return fmt.Sprintf("%s (recognized, long time ago)", f.Name)
		}
		return fmt.Sprintf("%s (recognized, last seen %s)", f.Name, dateStr)
	}
}

// formatAnonEntry — newcomer vs met-before.
//
//   "anon_73d0a4 (newcomer)"                          — < 3 min
//   "anon_73d0a4 (met before, last seen 2026-06-04)"  — ≥ 3 min
func formatAnonEntry(f FaceEntry) string {
	cfg := providerForDefaults.cfg
	if f.LastSeenAgoSec == nil {
		// Brand-new anon (just auto-enrolled this session) or no
		// session snapshot. Treat as newcomer.
		return fmt.Sprintf("%s (newcomer)", f.Name)
	}
	gap := *f.LastSeenAgoSec
	if gap < cfg.AnonNewcomerThrSec {
		return fmt.Sprintf("%s (newcomer)", f.Name)
	}
	dateStr := lastSeenDate(f.LastSeenISO)
	if dateStr == "" {
		return fmt.Sprintf("%s (met before)", f.Name)
	}
	return fmt.Sprintf("%s (met before, last seen %s)", f.Name, dateStr)
}

// lastSeenDate parses an ISO timestamp like "2026-03-05T14:30:00" into
// a human-friendly "2026-03-05" date. Returns "" on parse failure.
func lastSeenDate(isoPtr *string) string {
	if isoPtr == nil || *isoPtr == "" {
		return ""
	}
	iso := *isoPtr
	cutLen := len(iso)
	if cutLen > 19 {
		cutLen = 19
	}
	if t, err := time.Parse("2006-01-02T15:04:05", iso[:cutLen]); err == nil {
		return t.Format("2006-01-02")
	}
	// Fallback: take first 10 chars if they look like a date
	if len(iso) >= 10 && iso[4] == '-' && iso[7] == '-' {
		return iso[:10]
	}
	return ""
}