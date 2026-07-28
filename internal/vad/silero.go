// Package vad detects speech activity locally using the Silero VAD v5 ONNX
// model (https://huggingface.co/runanywhere/silero-vad-v5), so an ASR
// vendor's transcript-return latency can be measured against a
// locally-observed end-of-speech moment instead of the vendor's own
// (often later, and vendor-biased) speech_end event.
package vad

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// SampleRate is the sample rate Silero VAD v5 was trained and exported for.
const SampleRate = 16000

// FrameSamples is the fixed window size (32ms @ 16kHz) of *new* audio one
// Infer call consumes.
const FrameSamples = 512

const stateDim = 128

// contextSize is the number of trailing samples from the previous frame that
// Silero's own reference wrapper prepends to each new frame before running
// inference (see OnnxWrapper.__call__ in the upstream silero-vad repo).
// Without it the model's conv layers see a truncated receptive field on
// every call and speech probabilities stay near zero regardless of content.
const contextSize = 64

// modelInputSamples is the total sample count actually fed to the ONNX
// graph per call: contextSize carried over plus FrameSamples of new audio.
const modelInputSamples = contextSize + FrameSamples

var (
	envOnce sync.Once
	envErr  error
)

// ensureEnvironment initializes the process-wide onnxruntime environment
// exactly once, loading the shared library from libPath if given.
func ensureEnvironment(libPath string) error {
	envOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		envErr = ort.InitializeEnvironment()
	})
	return envErr
}

// Model wraps one loaded Silero VAD v5 ONNX session, its recurrent state,
// and the trailing-sample context carried between calls. It is not safe for
// concurrent use.
type Model struct {
	session *ort.DynamicAdvancedSession
	state   *ort.Tensor[float32]
	sr      *ort.Scalar[int64]
	context [contextSize]float32
}

// NewModel loads the Silero VAD v5 ONNX model at modelPath. libPath is the
// onnxruntime shared library to dlopen; see ResolveLibraryPath.
func NewModel(modelPath, libPath string) (*Model, error) {
	if err := ensureEnvironment(libPath); err != nil {
		return nil, fmt.Errorf("vad: init onnxruntime environment: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "sr", "state"},
		[]string{"output", "stateN"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("vad: load model %q: %w", modelPath, err)
	}

	state, err := ort.NewTensor(ort.NewShape(2, 1, stateDim), make([]float32, 2*stateDim))
	if err != nil {
		_ = session.Destroy()
		return nil, fmt.Errorf("vad: allocate state tensor: %w", err)
	}

	sr, err := ort.NewScalar(int64(SampleRate))
	if err != nil {
		_ = state.Destroy()
		_ = session.Destroy()
		return nil, fmt.Errorf("vad: allocate sample-rate tensor: %w", err)
	}

	return &Model{session: session, state: state, sr: sr}, nil
}

// Close releases the model's ONNX session and tensors.
func (m *Model) Close() error {
	_ = m.sr.Destroy()
	_ = m.state.Destroy()
	return m.session.Destroy()
}

// Reset zeroes the recurrent state and trailing context, starting a fresh
// utterance context.
func (m *Model) Reset() {
	data := m.state.GetData()
	for i := range data {
		data[i] = 0
	}
	m.context = [contextSize]float32{}
}

// Infer runs one inference step over exactly FrameSamples of *new* mono
// audio, normalized to [-1, 1], returning the model's speech probability for
// that frame. Internally, the trailing contextSize samples from the
// previous call are prepended before running (matching Silero's own
// reference wrapper), and the recurrent state carries forward to the next
// call.
func (m *Model) Infer(frame []float32) (float32, error) {
	if len(frame) != FrameSamples {
		return 0, fmt.Errorf("vad: frame must be %d samples, got %d", FrameSamples, len(frame))
	}

	var input [modelInputSamples]float32
	copy(input[:contextSize], m.context[:])
	copy(input[contextSize:], frame)

	in, err := ort.NewTensor(ort.NewShape(1, modelInputSamples), input[:])
	if err != nil {
		return 0, fmt.Errorf("vad: input tensor: %w", err)
	}
	defer func() { _ = in.Destroy() }()

	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return 0, fmt.Errorf("vad: output tensor: %w", err)
	}
	defer func() { _ = out.Destroy() }()

	newState, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, stateDim))
	if err != nil {
		return 0, fmt.Errorf("vad: state tensor: %w", err)
	}
	defer func() { _ = newState.Destroy() }()

	if err := m.session.Run([]ort.Value{in, m.sr, m.state}, []ort.Value{out, newState}); err != nil {
		return 0, fmt.Errorf("vad: run: %w", err)
	}

	copy(m.state.GetData(), newState.GetData())
	copy(m.context[:], input[modelInputSamples-contextSize:])
	return out.GetData()[0], nil
}
