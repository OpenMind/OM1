package autonomy

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestNormalizeAngle(t *testing.T) {
	cases := []struct {
		in, want float64
	}{
		{0, 0},
		{90, 90},
		{180, 180},
		{-180, -180},
		{181, -179},
		{270, -90},
		{-181, 179},
		{-270, 90},
	}
	for _, c := range cases {
		if got := normalizeAngle(c.in); got != c.want {
			t.Errorf("normalizeAngle(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestAngleGap(t *testing.T) {
	cases := []struct {
		current, target, want float64
	}{
		{10, 0, 10},
		{0, 10, -10},
		{170, -170, -20},
		{-170, 170, 20},
		{0, 0, 0},
	}
	for _, c := range cases {
		if got := angleGap(c.current, c.target); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("angleGap(%v, %v) = %v, want %v", c.current, c.target, got, c.want)
		}
	}
}

func TestSerializeTwist(t *testing.T) {
	payload := serializeTwist(0.5, 0, 0, 0, 0, -0.8)

	if len(payload) != 4+6*8 {
		t.Fatalf("twist length = %d, want %d", len(payload), 4+6*8)
	}

	if payload[0] != 0x00 || payload[1] != 0x01 || payload[2] != 0x00 || payload[3] != 0x00 {
		t.Errorf("unexpected CDR header: % x", payload[0:4])
	}

	readF64 := func(off int) float64 {
		return math.Float64frombits(binary.LittleEndian.Uint64(payload[off:]))
	}

	if got := readF64(4); got != 0.5 {
		t.Errorf("linear.x = %v, want 0.5", got)
	}
	if got := readF64(4 + 5*8); got != -0.8 {
		t.Errorf("angular.z = %v, want -0.8", got)
	}
}

func TestAIStatusRoundTrip(t *testing.T) {
	payload := serializeAIStatusResponse("frame-1", "req-42", aiCodeEnabled, "AI Control Enabled")

	req, err := deserializeAIStatusRequest(payload)
	if err != nil {
		t.Fatalf("deserialize: %v", err)
	}
	if req.frameID != "frame-1" {
		t.Errorf("frameID = %q, want %q", req.frameID, "frame-1")
	}
	if req.requestID != "req-42" {
		t.Errorf("requestID = %q, want %q", req.requestID, "req-42")
	}
	if req.code != aiCodeEnabled {
		t.Errorf("code = %d, want %d", req.code, aiCodeEnabled)
	}
}
