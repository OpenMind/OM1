// Package providers — FacePresenceProvider polls the face service's /who
// endpoint and renders snapshots into LLM-readable text.
//
// Greeting tiers for NAMED faces (by last-seen gap):
//   <1 day → "Welcome back", 1-6 days → "N days ago", ≥7 days → date.
// ANON faces: <3 min → newcomer, ≥3 min → met before.
// UNKNOWN faces (pre-enroll transients) are hidden from LLM text.
//
// "Last seen" gap is snapshotted at session start, not updated per-frame.
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

	"github.com/openmind/om1/internal/httpclient"
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
		cfg: cfg,
		client: &http.Client{
			Transport: httpclient.Default().Transport,
			Timeout:   cfg.Timeout,
		},
	}
}

// FaceEntry is one face in a /who snapshot. Pointer fields distinguish null from zero.
type FaceEntry struct {
	Name           string   `json:"name"` // "sean" / "anon_73d0a4" / "unknown"
	UUID           string   `json:"uuid"` // full 32-hex; "" for unknown
	Tier           string   `json:"tier"` // "confident" / "tentative" / "uncertain"
	Sim            float64  `json:"sim"`  // raw cosine sim
	TrackID        int      `json:"track_id"`
	Area           int      `json:"area,omitempty"`
	CreatedAgoSec  *float64 `json:"created_ago_sec"`   // UUID age in seconds; null for unknown
	LastSeenAgoSec *float64 `json:"last_seen_ago_sec"` // sticky session-start gap; null until first confident match
	LastSeenISO    *string  `json:"last_seen_iso"`     // ISO timestamp of previous sighting
}

// PresenceSnapshot is the parsed /who response.
// Closest* fields describe the largest-area face; new code should iterate Faces.
type PresenceSnapshot struct {
	OK       bool        `json:"ok"`
	Faces    []FaceEntry `json:"faces"`
	ServerTS float64     `json:"server_ts"`

	// Derived (populated by FetchSnapshot, not from JSON).
	ClosestName  string  `json:"-"`
	ClosestUUID  string  `json:"-"`
	ClosestTier  string  `json:"-"`
	ClosestSim   float64 `json:"-"`
	UnknownFaces int     `json:"-"`
}

// providerForDefaults holds thresholds used by ToText. Updated via SetDefaults.
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
// Populates derived Closest* and UnknownFaces fields before returning.
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
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return PresenceSnapshot{}, fmt.Errorf("/who body read: %w", err)
	}
	var snap PresenceSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return PresenceSnapshot{}, fmt.Errorf("/who decode: %w (body=%s)", err, string(data))
	}

	// Populate Closest* from the largest visible face with a UUID.
	// Anon names are left empty so the hook's fallback kicks in.
	var bestArea int
	for _, f := range snap.Faces {
		if f.Name == "unknown" || f.Name == "" {
			snap.UnknownFaces++
		}
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
// Returns "" when no actionable faces are present.
func (s *PresenceSnapshot) ToText() string {
	if s == nil || len(s.Faces) == 0 {
		return ""
	}

	// Sort by area desc.
	faces := make([]FaceEntry, len(s.Faces))
	copy(faces, s.Faces)
	sort.SliceStable(faces, func(i, j int) bool {
		if faces[i].Area != faces[j].Area {
			return faces[i].Area > faces[j].Area
		}
		return faces[i].TrackID < faces[j].TrackID
	})

	// Bucket into named/anon; drop unknown (transient, no UUID).
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

// formatNamedEntry — three-tier label based on last-seen gap.
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

// formatAnonEntry — newcomer (<3 min) vs met-before (≥3 min).
func formatAnonEntry(f FaceEntry) string {
	cfg := providerForDefaults.cfg
	if f.LastSeenAgoSec == nil {
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

// lastSeenDate extracts a "2006-01-02" date from an ISO timestamp.
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
