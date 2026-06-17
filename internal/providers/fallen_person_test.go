package providers

import (
	"math"
	"testing"
)

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
