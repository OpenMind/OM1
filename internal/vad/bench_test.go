package vad

import (
	"os"
	"testing"
)

// BenchmarkInferLocal measures per-frame inference cost running the ONNX
// graph locally on CPU, i.e. today's production path.
func BenchmarkInferLocal(b *testing.B) {
	libPath := ResolveLibraryPath(os.Getenv("OM1_TEST_ONNXRUNTIME_LIB"))
	if libPath == "" {
		b.Skip("onnxruntime shared library not found; set OM1_TEST_ONNXRUNTIME_LIB or run `make download-onnxruntime`")
	}
	modelPath := os.Getenv("OM1_TEST_VAD_MODEL")
	if modelPath == "" {
		modelPath = "models/silero_vad_v5.onnx"
	}
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("VAD model not found at %s; set OM1_TEST_VAD_MODEL or run `make download-vad-model`", modelPath)
	}

	model, err := NewModel(modelPath, libPath)
	if err != nil {
		b.Fatalf("NewModel: %v", err)
	}
	defer func() { _ = model.Close() }()

	benchmarkInfer(b, model)
}

// BenchmarkInferRemote measures per-frame inference cost going through the
// GPU service (see docker/Dockerfile.vad in OM1-modules). Point
// VAD_SERVICE_URL at a running instance, e.g.:
//
//	docker run --rm --runtime nvidia -p 8200:8200 vad-gpu-service:dev
//	VAD_SERVICE_URL=http://localhost:8200 go test -bench=InferRemote ./internal/vad/...
func BenchmarkInferRemote(b *testing.B) {
	url := os.Getenv("VAD_SERVICE_URL")
	if url == "" {
		b.Skip("VAD_SERVICE_URL not set; point it at a running vad-gpu-service (e.g. http://localhost:8200)")
	}
	benchmarkInfer(b, NewRemoteModel(url))
}

// benchmarkInfer drives FrameSamples-sized near-silent frames through model
// at the raw per-call rate, matching what the ASR hot path pays every 32ms
// per audio stream (FrameSamples/SampleRate seconds apart in production;
// the benchmark itself runs back-to-back with no sleep, to measure pure
// per-call cost rather than wall-clock stream time).
func benchmarkInfer(b *testing.B, model Inferer) {
	frame := make([]float32, FrameSamples)
	for i := range frame {
		frame[i] = 0.01 // near-silence, avoids any model-internal fast path
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Infer(frame); err != nil {
			b.Fatalf("Infer: %v", err)
		}
	}
}
