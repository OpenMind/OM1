package vad

import (
	"os"
	"testing"
)

// TestModelInferOnRealRuntime exercises the actual Silero VAD v5 ONNX model
// via onnxruntime. It's skipped unless both the onnxruntime shared library
// and the model file are available, since neither ships in the repo (see
// `make download-onnxruntime` and `make download-vad-model`).
func TestModelInferOnRealRuntime(t *testing.T) {
	libPath := ResolveLibraryPath(os.Getenv("OM1_TEST_ONNXRUNTIME_LIB"))
	if libPath == "" {
		t.Skip("onnxruntime shared library not found; set OM1_TEST_ONNXRUNTIME_LIB or run `make download-onnxruntime`")
	}
	modelPath := os.Getenv("OM1_TEST_VAD_MODEL")
	if modelPath == "" {
		modelPath = "models/silero_vad_v5.onnx"
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("VAD model not found at %s; set OM1_TEST_VAD_MODEL or run `make download-vad-model`", modelPath)
	}

	model, err := NewModel(modelPath, libPath)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	silence := make([]float32, FrameSamples)
	var last float32
	for i := 0; i < 5; i++ {
		prob, err := model.Infer(silence)
		if err != nil {
			t.Fatalf("Infer: %v", err)
		}
		last = prob
	}
	if last >= 0.5 {
		t.Errorf("expected low speech probability for silence, got %v", last)
	}
}
