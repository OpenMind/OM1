package providers

import (
	"math"
	"testing"
)

func TestFallenCache(t *testing.T) {
	p := &FallenPersonProvider{cfg: FallenPersonConfig{CacheFrames: 2}}

	present := FallenSnapshot{Present: true, Alert: true, Name: "wendy", NormErrX: 0.4, WidthFrac: 0.3}
	absent := FallenSnapshot{Present: false}

	// Fresh detection passes through and is cached.
	if got := p.cache(present); !got.Present || got.Cached {
		t.Fatalf("fresh frame: got Present=%v Cached=%v, want Present=true Cached=false", got.Present, got.Cached)
	}

	// First two dropouts reuse the cached target, marked Cached.
	for i := 1; i <= 2; i++ {
		got := p.cache(absent)
		if !got.Present || !got.Cached {
			t.Fatalf("dropout %d: got Present=%v Cached=%v, want Present=true Cached=true", i, got.Present, got.Cached)
		}
		if got.Name != "wendy" || got.NormErrX != 0.4 {
			t.Errorf("dropout %d: cached geometry not reused: %+v", i, got)
		}
	}

	// Third consecutive dropout exceeds CacheFrames: cache expires, empty passes through.
	if got := p.cache(absent); got.Present {
		t.Fatalf("after cache expiry: got Present=true, want false")
	}

	// A fresh detection re-primes the cache.
	if got := p.cache(present); !got.Present || got.Cached {
		t.Fatalf("re-prime: got Present=%v Cached=%v, want Present=true Cached=false", got.Present, got.Cached)
	}
	if got := p.cache(absent); !got.Cached {
		t.Errorf("re-prime dropout: want Cached=true, got %+v", got)
	}
}

func TestDeriveFallenGeometryFaceMatch(t *testing.T) {
	const eps = 1e-6
	// Body centered (center 320), face off to the right (center 480) → centering must
	// follow the face, distance must stay on the body width.
	raw := fallenResponse{
		Alert:   true,
		FrameHW: []float64{480, 640},
		FallenNowDetails: []FallenDetection{
			{Name: "wendy", Bbox: []float64{120, 200, 520, 360}, Confidence: 0.84}, // body, cx=320, w=400
		},
		Faces: []FaceDetection{
			{Name: "someone_else", Bbox: []float64{0, 0, 40, 40}},
			{Name: "WENDY", Bbox: []float64{440, 180, 520, 260}}, // case-insensitive, cx=480
		},
	}
	got := deriveFallenGeometry(raw)
	if got.FaceBbox == nil {
		t.Fatal("expected a matched face bbox")
	}
	if math.Abs(got.NormErrX-(480.0-320.0)/320.0) > eps {
		t.Errorf("NormErrX = %v, want face-based 0.5", got.NormErrX)
	}
	if got.HPos != "right" {
		t.Errorf("HPos = %q, want right (face position)", got.HPos)
	}
	if math.Abs(got.WidthFrac-400.0/640.0) > eps {
		t.Errorf("WidthFrac = %v, want body-based %v", got.WidthFrac, 400.0/640.0)
	}

	// "unknown" downed person: never face-matched, steered by body (cx=320 → center).
	rawUnknown := fallenResponse{
		Alert:            true,
		FrameHW:          []float64{480, 640},
		FallenNowDetails: []FallenDetection{{Name: "unknown", Bbox: []float64{120, 200, 520, 360}}},
		Faces:            []FaceDetection{{Name: "unknown", Bbox: []float64{600, 10, 639, 60}}},
	}
	gotU := deriveFallenGeometry(rawUnknown)
	if !gotU.Present {
		t.Fatal("unknown downed person should still be a present target")
	}
	if gotU.FaceBbox != nil {
		t.Error("unknown name must not face-match")
	}
	if gotU.HPos != "center" {
		t.Errorf("HPos = %q, want center (body-based)", gotU.HPos)
	}
}

func TestDeriveFallenGeometry(t *testing.T) {
	const eps = 1e-6

	tests := []struct {
		name          string
		raw           fallenResponse
		wantPresent   bool
		wantHPos      string
		wantNormErrX  float64
		wantWidthFrac float64
		wantName      string
	}{
		{
			name: "centered person at 640x480",
			raw: fallenResponse{
				Alert:   true,
				FrameHW: []float64{480, 640},
				FallenNowDetails: []FallenDetection{
					{Name: "wendy", Bbox: []float64{160, 200, 480, 300}, Confidence: 0.9},
				},
			},
			wantPresent:   true,
			wantHPos:      "center",
			wantNormErrX:  0, // center at 320 == frame center
			wantWidthFrac: 0.5,
			wantName:      "wendy",
		},
		{
			name: "person on the right",
			raw: fallenResponse{
				Alert:   true,
				FrameHW: []float64{480, 640},
				FallenNowDetails: []FallenDetection{
					{Name: "a", Bbox: []float64{500, 200, 620, 300}},
				},
			},
			wantPresent:   true,
			wantHPos:      "right",
			wantNormErrX:  (560.0 - 320.0) / 320.0, // +0.75
			wantWidthFrac: 120.0 / 640.0,
		},
		{
			name: "person on the left",
			raw: fallenResponse{
				Alert:   true,
				FrameHW: []float64{480, 640},
				FallenNowDetails: []FallenDetection{
					{Name: "a", Bbox: []float64{20, 200, 140, 300}},
				},
			},
			wantPresent:   true,
			wantHPos:      "left",
			wantNormErrX:  (80.0 - 320.0) / 320.0, // -0.75
			wantWidthFrac: 120.0 / 640.0,
		},
		{
			name: "frame_hw absent falls back to 640 wide",
			raw: fallenResponse{
				Alert: true,
				FallenNowDetails: []FallenDetection{
					{Name: "a", Bbox: []float64{288, 200, 352, 300}},
				},
			},
			wantPresent:   true,
			wantHPos:      "center",
			wantNormErrX:  0,
			wantWidthFrac: 64.0 / 640.0,
		},
		{
			name: "picks largest-area detail",
			raw: fallenResponse{
				Alert:   true,
				FrameHW: []float64{480, 640},
				FallenNowDetails: []FallenDetection{
					{Name: "small", Bbox: []float64{0, 0, 20, 20}},
					{Name: "big", Bbox: []float64{100, 100, 540, 380}},
				},
			},
			wantPresent: true,
			wantName:    "big",
			wantHPos:    "center",
		},
		{
			name:        "no details, not present",
			raw:         fallenResponse{Alert: false},
			wantPresent: false,
		},
		{
			name: "degenerate bbox ignored",
			raw: fallenResponse{
				Alert:   true,
				FrameHW: []float64{480, 640},
				FallenNowDetails: []FallenDetection{
					{Name: "bad", Bbox: []float64{300, 300, 300, 300}},
				},
			},
			wantPresent: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deriveFallenGeometry(tt.raw)
			if got.Present != tt.wantPresent {
				t.Fatalf("Present = %v, want %v", got.Present, tt.wantPresent)
			}
			if !tt.wantPresent {
				return
			}
			if tt.wantName != "" && got.Name != tt.wantName {
				t.Errorf("Name = %q, want %q", got.Name, tt.wantName)
			}
			if tt.wantHPos != "" && got.HPos != tt.wantHPos {
				t.Errorf("HPos = %q, want %q", got.HPos, tt.wantHPos)
			}
			if tt.wantWidthFrac != 0 && math.Abs(got.WidthFrac-tt.wantWidthFrac) > eps {
				t.Errorf("WidthFrac = %v, want %v", got.WidthFrac, tt.wantWidthFrac)
			}
			if math.Abs(got.NormErrX-tt.wantNormErrX) > eps {
				t.Errorf("NormErrX = %v, want %v", got.NormErrX, tt.wantNormErrX)
			}
		})
	}
}
