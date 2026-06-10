package qr_scanner_test

import (
	"testing"

	"github.com/openmind/om1/plugins/inputs/qr_scanner"
)

func TestParseLumaCheckinURL(t *testing.T) {
	cases := []struct {
		name      string
		input     string
		wantEvent string
		wantPK    string
		wantOK    bool
	}{
		{
			name:      "happy path luma.com",
			input:     "https://luma.com/check-in/evt-abc?pk=g-12345",
			wantEvent: "evt-abc",
			wantPK:    "g-12345",
			wantOK:    true,
		},
		{
			name:      "lu.ma short host",
			input:     "https://lu.ma/check-in/evt-abc?pk=tk_xyz",
			wantEvent: "evt-abc",
			wantPK:    "tk_xyz",
			wantOK:    true,
		},
		{
			name:      "www.luma.com",
			input:     "https://www.luma.com/check-in/evt-abc?pk=g-12345",
			wantEvent: "evt-abc",
			wantPK:    "g-12345",
			wantOK:    true,
		},
		{
			name:      "trailing slash on path",
			input:     "https://luma.com/check-in/evt-abc/?pk=g-12345",
			wantEvent: "evt-abc",
			wantPK:    "g-12345",
			wantOK:    true,
		},
		{
			name:      "http scheme accepted",
			input:     "http://luma.com/check-in/evt-abc?pk=g-12345",
			wantEvent: "evt-abc",
			wantPK:    "g-12345",
			wantOK:    true,
		},
		{
			name:      "percent-encoded pk",
			input:     "https://luma.com/check-in/evt-abc?pk=g%2D12345",
			wantEvent: "evt-abc",
			wantPK:    "g-12345",
			wantOK:    true,
		},
		{
			name:   "missing pk",
			input:  "https://luma.com/check-in/evt-abc",
			wantOK: false,
		},
		{
			name:   "empty pk",
			input:  "https://luma.com/check-in/evt-abc?pk=",
			wantOK: false,
		},
		{
			name:   "wrong host",
			input:  "https://example.com/check-in/evt-abc?pk=g-12345",
			wantOK: false,
		},
		{
			name:   "wrong path",
			input:  "https://luma.com/event/evt-abc?pk=g-12345",
			wantOK: false,
		},
		{
			name:   "extra path segment",
			input:  "https://luma.com/check-in/evt-abc/extra?pk=g-12345",
			wantOK: false,
		},
		{
			name:   "missing event id",
			input:  "https://luma.com/check-in/?pk=g-12345",
			wantOK: false,
		},
		{
			name:   "non-url garbage",
			input:  "hello world",
			wantOK: false,
		},
		{
			name:   "ftp scheme rejected",
			input:  "ftp://luma.com/check-in/evt-abc?pk=g-12345",
			wantOK: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotEvent, gotPK, gotOK := qr_scanner.ParseLumaCheckinURL(tc.input)
			if gotOK != tc.wantOK {
				t.Fatalf("ok mismatch: got %v want %v", gotOK, tc.wantOK)
			}
			if !tc.wantOK {
				return
			}
			if gotEvent != tc.wantEvent {
				t.Errorf("event: got %q want %q", gotEvent, tc.wantEvent)
			}
			if gotPK != tc.wantPK {
				t.Errorf("pk: got %q want %q", gotPK, tc.wantPK)
			}
		})
	}
}
