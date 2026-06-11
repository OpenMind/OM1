package config

import "testing"

func TestGetRuntimeVersion(t *testing.T) {
	if GetRuntimeVersion() != LatestRuntimeVersion {
		t.Fatalf("expected %q, got %q", LatestRuntimeVersion, GetRuntimeVersion())
	}
	if LatestRuntimeVersion != "v1.1.0" {
		t.Fatalf("expected runtime version v1.1.0, got %q", LatestRuntimeVersion)
	}
}

func TestIsVersionSupported(t *testing.T) {
	tests := []struct {
		name      string
		version   string
		supported bool
		wantErr   bool
	}{
		{"empty", "", false, true},
		{"exact", LatestRuntimeVersion, true, false},
		{"no v prefix", "1.1.0", true, false},
		{"patch differs", "v1.1.99", true, false},
		{"minor differs warns", "v1.0.0", true, false},
		{"major mismatch", "v2.0.0", false, true},
		{"invalid", "abc", false, true},
		{"too many parts", "1.1.0.0", false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := IsVersionSupported(tt.version)
			if (err != nil) != tt.wantErr {
				t.Fatalf("wantErr=%v, got err=%v", tt.wantErr, err)
			}
			if got != tt.supported {
				t.Fatalf("expected supported=%v, got %v", tt.supported, got)
			}
		})
	}
}

func TestVerifyRuntimeVersion(t *testing.T) {
	if _, err := VerifyRuntimeVersion("v1.1.0", "test_config"); err != nil {
		t.Fatalf("expected compatible, got %v", err)
	}

	if _, err := VerifyRuntimeVersion("v2.0.0", "test_config"); err == nil {
		t.Fatal("expected error for major version mismatch")
	}

	if _, err := VerifyRuntimeVersion("", ""); err == nil {
		t.Fatal("expected error for empty version")
	}
}
