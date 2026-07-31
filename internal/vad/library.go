package vad

import (
	"os"
	"path/filepath"
)

var defaultLibraryCandidates = []string{
	".onnxruntime/lib/libonnxruntime.so",
	".onnxruntime/lib/libonnxruntime.dylib",
	"/usr/local/lib/libonnxruntime.so",
	"/usr/lib/libonnxruntime.so",
}

// ResolveLibraryPath returns the onnxruntime shared library to dlopen
func ResolveLibraryPath(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if v := os.Getenv("OM1_ONNXRUNTIME_LIB"); v != "" {
		return v
	}
	for _, c := range defaultLibraryCandidates {
		if _, err := os.Stat(c); err != nil {
			continue
		}
		if abs, err := filepath.Abs(c); err == nil {
			return abs
		}
		return c
	}
	return ""
}
