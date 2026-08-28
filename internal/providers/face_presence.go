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

const (
	facePresenceDefaultBaseURL                = "http://127.0.0.1:6793"
	facePresenceDefaultRecentSec              = 1.0
	facePresenceDefaultTimeout                = 2 * time.Second
	facePresenceDefaultAnonNewcomerThrSec     = 180.0     // 3 min
	facePresenceDefaultNamedRecentThrSec      = 86_400.0  // 1 day
	facePresenceDefaultNamedLongAbsenceThrSec = 604_800.0 // 7 days
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
		cfg.BaseURL = facePresenceDefaultBaseURL
	}
	if cfg.RecentSec <= 0 {
		cfg.RecentSec = facePresenceDefaultRecentSec
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = facePresenceDefaultTimeout
	}
	if cfg.AnonNewcomerThrSec <= 0 {
		cfg.AnonNewcomerThrSec = facePresenceDefaultAnonNewcomerThrSec
	}
	if cfg.NamedRecentThrSec <= 0 {
		cfg.NamedRecentThrSec = facePresenceDefaultNamedRecentThrSec
	}
	if cfg.NamedLongAbsenceThrSec <= 0 {
		cfg.NamedLongAbsenceThrSec = facePresenceDefaultNamedLongAbsenceThrSec
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
	Enrolling      bool     `json:"enrolling"`
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
		AnonNewcomerThrSec:     facePresenceDefaultAnonNewcomerThrSec,
		NamedRecentThrSec:      facePresenceDefaultNamedRecentThrSec,
		NamedLongAbsenceThrSec: facePresenceDefaultNamedLongAbsenceThrSec,
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
	return s.toTextWithSpeaker(Speaker().Latest(), Speaker().Available())
}

// toTextWithSpeaker renders the line against a given speaker verdict.
//
// Split from ToText so the rendering can be exercised for a speaker that is
// present, absent, unresolved, or off-screen without standing up an HTTP
// server and driving the singleton into each of those states.
func (s *PresenceSnapshot) toTextWithSpeaker(spk *SpeakerResult, available bool) string {
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

	speakingTrack := -1
	if spk.Identified() {
		speakingTrack = spk.TrackID
	}

	var kept []*FaceEntry
	for i := range faces {
		f := &faces[i]
		if f.Name == "unknown" || f.Name == "" {
			if f.TrackID != speakingTrack || speakingTrack < 0 {
				continue
			}
		}
		kept = append(kept, f)
	}

	if len(kept) == 0 {
		return ""
	}

	closest := kept[0]

	descriptor := fmt.Sprintf("FacePresence: %d face", len(kept))
	if len(kept) != 1 {
		descriptor += "s"
	}
	// Tell the model the list is ordered by proximity and what that implies.
	descriptor += " (nearest first; nearest face is closest to the camera and most likely addressing the robot)"

	var parts []string
	var speakerEntry string
	speakerEnrolling := false
	speakerUUID := ""
	for _, f := range kept {
		var entry string
		switch {
		case f.Name == "unknown" || f.Name == "":
			entry = "an unrecognised person"
		case strings.HasPrefix(f.Name, "anon_"):
			entry = formatAnonEntry(*f)
		default:
			entry = formatNamedEntry(*f)
		}
		switch {
		case speakingTrack >= 0 && f.TrackID == speakingTrack:
			entry += " [SPEAKING NOW]"
			speakerEntry = entry
			speakerEnrolling = f.Enrolling
			// The face's OWN uuid, not the speaker verdict's.
			//
			// /speaking reports a uuid only for a track it matched
			// confidently at that instant, so it comes back null for
			// somebody the video is plainly labelling anon_xxxx. Testing
			// the verdict's copy therefore concluded "no identity" about a
			// person who visibly has one, and the robot asked them to step
			// closer -- repeatedly, and pointlessly, since nothing about
			// moving would have changed an identity they already had.
			speakerUUID = f.UUID
		case speakingTrack >= 0:
			// A measured speaker exists and it is not this face. Saying
			// nothing here would leave the model to fall back on the
			// proximity hint in the descriptor and contradict the
			// measurement.
			entry += " [not speaking]"
		case f == closest:
			entry += " [closest, likely speaking — GUESS, no speech detection]"
		}
		parts = append(parts, entry)
	}

	line := fmt.Sprintf("%s — %s", descriptor, strings.Join(parts, ", "))
	if speakerUUID == "" {
		speakerUUID = spk.identityUUID()
	} else {
		// Hand it back so the rename path sees the identity too, instead of
		// re-deriving it from a track lookup that misses the same faces.
		Speaker().NoteIdentity(speakingTrack, speakerUUID)
	}
	return line + speakerSuffix(
		spk, available, speakingTrack, speakerEntry, speakerEnrolling, speakerUUID)
}

func speakerSuffix(
	spk *SpeakerResult, available bool, speakingTrack int,
	speakerEntry string, speakerEnrolling bool, speakerUUID string,
) string {
	if !available {
		return "\nSpeaker: unknown (no active-speaker detection running; " +
			"do NOT assume the nearest face is the one talking)"
	}
	if spk == nil {
		if Speaker().Pending() {
			// Distinct from "nobody spoke": the answer is on its way and
			// simply did not beat the prompt. Saying so stops the model
			// treating the gap as evidence that nobody was talking.
			return "\nSpeaker: still being resolved for this utterance — " +
				"do not attribute it to anyone yet"
		}
		return "\nSpeaker: unknown (no utterance resolved yet)"
	}
	if speakingTrack < 0 {
		return "\nSpeaker: unknown (nobody scored as speaking over the last utterance)"
	}
	if speakerUUID == "" && !speakerEnrolling {
		// They spoke, the robot heard them, and there is nothing to hang a
		// name on -- the face is too far away, too small or too turned away
		// to clear the enrolment gate, and standing there will not change
		// that. Worth saying, because the fix is something a person can do
		// in one step; worth saying ONCE, because it is a favour to ask and
		// not a requirement to meet.
		return fmt.Sprintf(
			"\nSpeaker: %s (track %d, confidence %.2f) — their face cannot be "+
				"enrolled from where they are, so you cannot save a name for "+
				"them yet. You may invite them once to come closer or face you; "+
				"if they do not, let it go and keep talking normally.",
			speakerName(spk), spk.TrackID, spk.Score)
	}
	if speakerEntry == "" {
		// Resolved to a track that is no longer among the visible faces --
		// they turned away or the track was dropped between the utterance
		// and this poll.
		return fmt.Sprintf(
			"\nSpeaker: %s (track %d, confidence %.2f) — no longer visible",
			speakerName(spk), spk.TrackID, spk.Score)
	}
	return fmt.Sprintf(
		"\nSpeaker: %s (track %d, confidence %.2f) — attribute what was just said to THIS person",
		speakerName(spk), spk.TrackID, spk.Score)
}

func (s *PresenceSnapshot) AttributedUser() (uuid string, name string, measured bool) {
	if s == nil {
		return "", "", false
	}
	if spk := Speaker().Latest(); spk.Identified() {
		for i := range s.Faces {
			f := &s.Faces[i]
			if f.TrackID != spk.TrackID || f.UUID == "" {
				continue
			}
			n := f.Name
			if strings.HasPrefix(n, "anon_") || n == "unknown" {
				n = ""
			}
			return f.UUID, n, true
		}
		// Measured, but that track is not in this snapshot. Prefer the
		// speaker's own copy of the identity over silently falling back to
		// whoever is nearest -- they are still the one who spoke.
		if spk.UUID != "" {
			n := spk.Name
			if strings.HasPrefix(n, "anon_") || n == "unknown" {
				n = ""
			}
			return spk.UUID, n, true
		}
	}
	return s.ClosestUUID, s.ClosestName, false
}

// speakerName renders the speaker's identity, keeping "unnamed" distinct from
// "unrecognised": the first can be given a name, the second cannot yet.
func speakerName(spk *SpeakerResult) string {
	switch {
	case spk.Name != "" && !strings.HasPrefix(spk.Name, "anon_") && spk.Name != "unknown":
		return spk.Name
	case spk.UUID != "":
		return "an enrolled but unnamed person"
	default:
		return "an unrecognised person"
	}
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
