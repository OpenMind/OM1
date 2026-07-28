package vad

import (
	"os"
	"path/filepath"
)

// defaultLibraryCandidates are searched, in order, when no explicit shared
// library path is configured. ".onnxruntime" matches the layout produced by
// `make download-onnxruntime`.
var defaultLibraryCandidates = []string{
	".onnxruntime/lib/libonnxruntime.so",
	".onnxruntime/lib/libonnxruntime.dylib",
	"/usr/local/lib/libonnxruntime.so",
	"/usr/lib/libonnxruntime.so",
}

// ResolveLibraryPath returns the onnxruntime shared library to dlopen:
// explicit if non-empty, else $OM1_ONNXRUNTIME_LIB, else the first existing
// default candidate, else "" (onnxruntime_go then falls back to searching
// the system loader path for "onnxruntime.so").
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
