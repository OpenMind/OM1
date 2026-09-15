package inputs

import (
	"encoding/base64"
	"testing"
)

func TestDecodeFrame(t *testing.T) {
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10}
	png := []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A}

	tests := []struct {
		name    string
		input   string
		want    []byte
		wantExt string
		wantErr bool
	}{
		{
			name:    "raw base64 jpeg",
			input:   base64.StdEncoding.EncodeToString(jpeg),
			want:    jpeg,
			wantExt: ".jpg",
		},
		{
			name:    "raw base64 png detected by magic",
			input:   base64.StdEncoding.EncodeToString(png),
			want:    png,
			wantExt: ".png",
		},
		{
			name:    "data-url jpeg prefix stripped",
			input:   "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(jpeg),
			want:    jpeg,
			wantExt: ".jpg",
		},
		{
			name:    "data-url png prefix stripped",
			input:   "data:image/png;base64," + base64.StdEncoding.EncodeToString(png),
			want:    png,
			wantExt: ".png",
		},
		{
			name:    "invalid base64 errors",
			input:   "not!base64!!",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ext, err := decodeFrame(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ext != tt.wantExt {
				t.Errorf("ext = %q, want %q", ext, tt.wantExt)
			}
			if string(got) != string(tt.want) {
				t.Errorf("bytes = % x, want % x", got, tt.want)
			}
		})
	}
}
