package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

// faceInfo is a single face entry in the /who response.
type faceInfo struct {
	Name string  `json:"name"`
	Area float64 `json:"area"`
}

// whoResponse is the body returned by POST {base_url}/who.
type whoResponse struct {
	Faces    []faceInfo `json:"faces"`
	ServerTS float64    `json:"server_ts"`
}

// PresenceSnapshot is the canonical record derived from a /who response.
type PresenceSnapshot struct {
	// Timestamp is the server timestamp in UNIX epoch seconds (falls back to local time if missing).
	Timestamp float64
	// Names holds the known identities present (deduplicated, ordered by face area descending).
	Names []string
	// UnknownFaces is the count of unknown faces present.
	UnknownFaces int
	// ClosestName is the name of the person with the largest face area, or "unknown" if unavailable.
	ClosestName string
}

// ToText produces a concise, natural sentence describing who is in view,
// handling any number of known people and unknown faces.
//
// Examples:
//   - names=["wendy"], unknown=0      -> "In Camera View: 1 known (wendy). Closest: wendy."
//   - names=["wendy","alice"], unk=2  -> "In Camera View: 2 known (wendy and alice) and 2 unknown faces. ..."
//   - names=[], unknown=1             -> "In Camera View: 1 unknown face. Closest: unknown."
//   - names=[], unknown=0             -> "No one in view."
func (s PresenceSnapshot) ToText() string {
	seen := make(map[string]struct{})
	var valid []string
	for _, name := range s.Names {
		cleaned := strings.TrimSpace(name)
		if cleaned == "" || strings.EqualFold(cleaned, "unknown") {
			continue
		}
		if _, ok := seen[cleaned]; ok {
			continue
		}
		seen[cleaned] = struct{}{}
		valid = append(valid, cleaned)
	}

	knownCount := len(valid)
	unknownCount := s.UnknownFaces
	if unknownCount < 0 {
		unknownCount = 0
	}

	if knownCount == 0 && unknownCount == 0 {
		return "No one in view."
	}

	var parts []string
	if knownCount > 0 {
		parts = append(parts, fmt.Sprintf("%d known (%s)", knownCount, joinNames(valid)))
	}
	if unknownCount > 0 {
		suffix := "s"
		if unknownCount == 1 {
			suffix = ""
		}
		parts = append(parts, fmt.Sprintf("%d unknown face%s", unknownCount, suffix))
	}

	result := "In Camera View: " + strings.Join(parts, " and ") + "."
	if s.ClosestName != "" {
		result += fmt.Sprintf(" Closest: %s.", s.ClosestName)
	}
	return result
}

// joinNames joins a list of names into a human-readable string with proper conjunctions.
func joinNames(names []string) string {
	switch len(names) {
	case 0:
		return ""
	case 1:
		return names[0]
	case 2:
		return names[0] + " and " + names[1]
	default:
		return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
	}
}

// FacePresenceConfig configures a FacePresenceProvider.
type FacePresenceConfig struct {
	// BaseURL is the base HTTP URL of the face stream API (e.g. "http://127.0.0.1:6793").
	BaseURL string
	// RecentSec is the lookback window passed to /who (seconds of presence history).
	RecentSec float64
	// Timeout bounds each HTTP request.
	Timeout time.Duration
	// MinFaceArea is the minimum face area (in pixels) for a detection to be counted.
	MinFaceArea float64
}

// FacePresenceProvider fetches face-presence snapshots from a face HTTP service.
type FacePresenceProvider struct {
	baseURL     string
	recentSec   float64
	timeout     time.Duration
	minFaceArea float64
	client      *http.Client
}

// NewFacePresenceProvider constructs a FacePresenceProvider from cfg, applying
// sane defaults for any zero-valued field.
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
	if cfg.MinFaceArea <= 0 {
		cfg.MinFaceArea = 500
	}
	return &FacePresenceProvider{
		baseURL:     strings.TrimRight(cfg.BaseURL, "/"),
		recentSec:   cfg.RecentSec,
		timeout:     cfg.Timeout,
		minFaceArea: cfg.MinFaceArea,
		client:      httpclient.Default(),
	}
}

// FetchSnapshot POSTs to {base_url}/who with the configured lookback window and
// builds a PresenceSnapshot. Faces smaller than MinFaceArea are ignored; names
// are deduplicated and ordered by descending face area.
func (p *FacePresenceProvider) FetchSnapshot(ctx context.Context) (PresenceSnapshot, error) {
	body, err := json.Marshal(map[string]any{"recent_sec": p.recentSec})
	if err != nil {
		return PresenceSnapshot{}, err
	}

	reqCtx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, p.baseURL+"/who", bytes.NewReader(body))
	if err != nil {
		return PresenceSnapshot{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return PresenceSnapshot{}, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return PresenceSnapshot{}, fmt.Errorf("face presence: /who returned status %d", resp.StatusCode)
	}

	var data whoResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return PresenceSnapshot{}, err
	}

	// Keep only sufficiently large faces, then order by descending area so the
	// first element is the face closest to the camera.
	var filtered []faceInfo
	for _, f := range data.Faces {
		if f.Area >= p.minFaceArea {
			filtered = append(filtered, f)
		}
	}
	sorted := make([]faceInfo, len(filtered))
	copy(sorted, filtered)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Area > sorted[j].Area })

	closestName := "unknown"
	if len(sorted) > 0 {
		if n := strings.TrimSpace(sorted[0].Name); n != "" && !strings.EqualFold(n, "unknown") {
			closestName = n
		}
	}

	seen := make(map[string]struct{})
	var names []string
	for _, f := range sorted {
		n := strings.TrimSpace(f.Name)
		if n == "" || strings.EqualFold(n, "unknown") {
			continue
		}
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		names = append(names, n)
	}

	unknownCount := 0
	for _, f := range filtered {
		if strings.EqualFold(strings.TrimSpace(f.Name), "unknown") {
			unknownCount++
		}
	}

	timestamp := data.ServerTS
	if timestamp == 0 {
		timestamp = float64(time.Now().UnixNano()) / 1e9
	}

	return PresenceSnapshot{
		Timestamp:    timestamp,
		Names:        names,
		UnknownFaces: unknownCount,
		ClosestName:  closestName,
	}, nil
}
