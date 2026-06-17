package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

const (
	fallenPersonDefaultBaseURL = facePresenceDefaultBaseURL // shared perception service
	fallenPersonDefaultPath    = "/who"
	fallenPersonDefaultTimeout = 2 * time.Second

	// fallenDefaultFrameW is the frame width assumed when the endpoint omits frame_hw.
	fallenDefaultFrameW = 640.0

	// DefaultLockWidthFrac is the bbox-width fraction of the frame at which the robot
	// is considered close enough to a downed person to lock on and hold. A lying body
	// spans the frame horizontally, so width grows as the robot approaches.
	DefaultLockWidthFrac = 0.7

	// DefaultCenterTol is the |normalized horizontal offset| within which the target is
	// treated as centered (≈ the middle third of the frame, matching the endpoint convention).
	DefaultCenterTol = 0.33
)

// FallenPersonConfig configures the HTTP poller for the fallen-person endpoint.
type FallenPersonConfig struct {
	BaseURL string
	Path    string
	Timeout time.Duration

	// CacheFrames is how many consecutive frames without a fresh bbox keep reusing
	// the last known target, so brief detector dropouts don't lose the person.
	CacheFrames int
}

// FallenDetection mirrors one entry of the endpoint's fallen_now_details array.
type FallenDetection struct {
	Name       string    `json:"name"`
	Bbox       []float64 `json:"bbox"` // [x1, y1, x2, y2] in frame pixels, origin top-left
	Confidence float64   `json:"confidence"`
}

// FaceDetection mirrors one entry of the endpoint's faces array. A face bbox is the
// head, a steadier aim point than the sprawling body box, so it is preferred for
// horizontal centering when its name matches a downed person.
type FaceDetection struct {
	Name string    `json:"name"`
	Bbox []float64 `json:"bbox"` // [x1, y1, x2, y2] in frame pixels, origin top-left
}

// fallenResponse is the subset of the /who response this provider consumes.
type fallenResponse struct {
	Alert            bool              `json:"alert"`
	FallenNowDetails []FallenDetection `json:"fallen_now_details"`
	Faces            []FaceDetection   `json:"faces"`
	FrameHW          []float64         `json:"frame_hw"` // [height, width]
	FrameB64         string            `json:"frame_b64"`
}

// FallenSnapshot is the geometry derived from the closest downed person, computed
// against the frame size reported by the endpoint.
type FallenSnapshot struct {
	Alert bool // endpoint-level alert flag

	// Present is true when a usable target bbox was found.
	Present    bool
	Name       string
	Confidence float64

	// NormErrX is the target's horizontal offset from frame center, normalized to
	// [-1, 1]: positive means the person is to the right of center, negative to the left.
	NormErrX float64

	// WidthFrac is the bbox width as a fraction of frame width — the distance proxy.
	WidthFrac float64

	// HPos is the coarse band ("left", "center", "right") using the endpoint's thirds rule.
	HPos string

	// Cached is true when this geometry was reused from a previous frame because the
	// current frame had no fresh bbox (a detector dropout within CacheFrames).
	Cached bool

	// BodyBbox is the chosen downed person's full-body bbox (the distance source).
	BodyBbox []float64
	// FaceBbox is that person's matched face bbox, or nil if no face matched; when set
	// it is the source of the horizontal centering (NormErrX/HPos).
	FaceBbox []float64

	// Detections is every downed-person detection from this response (for debug dumps).
	Detections []FallenDetection
	// FrameW is the frame width used for the geometry (from frame_hw, else the default).
	FrameW float64
	// FrameB64 is the raw base64 (or data-URL) frame image, decoded only when dumping.
	FrameB64 string
}

// FallenPersonProvider polls the endpoint and renders downed-person geometry.
type FallenPersonProvider struct {
	cfg    FallenPersonConfig
	client *http.Client

	// Cache of the last frame that had a fresh target, reused for up to cfg.CacheFrames
	// dropouts. Accessed only from the single poll goroutine, but guarded for safety.
	mu        sync.Mutex
	cached    FallenSnapshot
	hasCached bool
	missCount int
}

// NewFallenPersonProvider constructs a provider with defaults filled in.
func NewFallenPersonProvider(cfg FallenPersonConfig) *FallenPersonProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = fallenPersonDefaultBaseURL
	}
	if cfg.Path == "" {
		cfg.Path = fallenPersonDefaultPath
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = fallenPersonDefaultTimeout
	}
	return &FallenPersonProvider{
		cfg: cfg,
		client: &http.Client{
			Transport: httpclient.Default().Transport,
			Timeout:   cfg.Timeout,
		},
	}
}

// FetchSnapshot POSTs to the configured path and returns the derived geometry.
func (p *FallenPersonProvider) FetchSnapshot(ctx context.Context) (FallenSnapshot, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", p.cfg.BaseURL+p.cfg.Path, bytes.NewReader([]byte("{}")))
	if err != nil {
		return FallenSnapshot{}, fmt.Errorf("build fallen request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return FallenSnapshot{}, fmt.Errorf("fallen request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return FallenSnapshot{}, fmt.Errorf("fallen body read: %w", err)
	}

	var raw fallenResponse
	if err := json.Unmarshal(data, &raw); err != nil {
		return FallenSnapshot{}, fmt.Errorf("fallen decode: %w (body=%s)", err, string(data))
	}

	return p.cache(deriveFallenGeometry(raw)), nil
}

// cache stabilizes targeting across detector dropouts: when the current frame has a
// fresh target it is cached and returned as-is; when it has none, the last cached
// target is reused (with the live frame image, marked Cached) for up to cfg.CacheFrames
// consecutive frames, after which the cache expires and the empty result passes through
// so the alert can clear.
func (p *FallenPersonProvider) cache(snap FallenSnapshot) FallenSnapshot {
	p.mu.Lock()
	defer p.mu.Unlock()

	if snap.Present {
		p.cached = snap
		p.hasCached = true
		p.missCount = 0
		return snap
	}

	if p.hasCached && p.missCount < p.cfg.CacheFrames {
		p.missCount++
		out := p.cached
		out.Cached = true
		// Keep the live frame so debug dumps show what the camera sees right now,
		// even though the steering geometry is reused from the last detection.
		out.FrameB64 = snap.FrameB64
		out.Detections = snap.Detections
		if snap.FrameW > 0 {
			out.FrameW = snap.FrameW
		}
		return out
	}

	p.hasCached = false
	return snap
}

// deriveFallenGeometry picks the closest (largest-area) downed person and computes
// the centering/distance geometry. It is pure so it can be unit-tested directly.
func deriveFallenGeometry(raw fallenResponse) FallenSnapshot {
	// frame_hw is [height, width]; only the width matters for horizontal centering
	// and the width-fraction distance proxy.
	frameW := fallenDefaultFrameW
	if len(raw.FrameHW) == 2 && raw.FrameHW[1] > 0 {
		frameW = raw.FrameHW[1]
	}

	snap := FallenSnapshot{
		Alert:      raw.Alert,
		Detections: raw.FallenNowDetails,
		FrameW:     frameW,
		FrameB64:   raw.FrameB64,
	}

	var (
		best     FallenDetection
		bestArea float64
		found    bool
	)
	for _, d := range raw.FallenNowDetails {
		if len(d.Bbox) != 4 {
			continue
		}
		w := d.Bbox[2] - d.Bbox[0]
		h := d.Bbox[3] - d.Bbox[1]
		if w <= 0 || h <= 0 {
			continue
		}
		if area := w * h; area > bestArea {
			bestArea = area
			best = d
			found = true
		}
	}
	if !found {
		return snap
	}

	snap.Present = true
	snap.Name = best.Name
	snap.Confidence = best.Confidence
	snap.BodyBbox = best.Bbox

	// Distance always comes from the full-body width — a face is too small to gauge it.
	snap.WidthFrac = (best.Bbox[2] - best.Bbox[0]) / frameW

	// Centering prefers the matched face (the head); fall back to the body center.
	// "unknown" downed people are still approached, just steered by their body box.
	centerBbox := best.Bbox
	if face, ok := matchFace(raw.Faces, best.Name); ok {
		snap.FaceBbox = face.Bbox
		centerBbox = face.Bbox
	}

	cx := (centerBbox[0] + centerBbox[2]) / 2
	snap.NormErrX = (cx - frameW/2) / (frameW / 2)

	switch {
	case cx < frameW/3:
		snap.HPos = "left"
	case cx > 2*frameW/3:
		snap.HPos = "right"
	default:
		snap.HPos = "center"
	}

	return snap
}

// matchFace returns the face whose name equals the downed person's, used to steer by
// the head. "unknown" (or empty) names are never matched, so they fall back to body.
func matchFace(faces []FaceDetection, name string) (FaceDetection, bool) {
	if name == "" || strings.EqualFold(name, "unknown") {
		return FaceDetection{}, false
	}
	for _, f := range faces {
		if len(f.Bbox) == 4 && strings.EqualFold(f.Name, name) {
			return f, true
		}
	}
	return FaceDetection{}, false
}
