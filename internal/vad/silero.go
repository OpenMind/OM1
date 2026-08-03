// Package vad detects speech activity locally via the Silero VAD v5 ONNX
// model (https://huggingface.co/runanywhere/silero-vad-v5), so ASR latency
// can be measured against a local end-of-speech moment instead of a
// vendor's own (later, vendor-biased) speech_end event.
package vad

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// SampleRate is the sample rate Silero VAD v5 was trained and exported for.
const SampleRate = 16000

// FrameSamples is the fixed window size (32ms @ 16kHz) of new audio per call.
const FrameSamples = 512

const stateDim = 128

// Number of trailing samples that Silero prepends to new frames
const contextSize = 64

const modelInputSamples = contextSize + FrameSamples

var (
	envOnce sync.Once
	envErr  error
)

// initializes the process-wide onnxruntime environment
func ensureEnvironment(libPath string) error {
	envOnce.Do(func() {
		if libPath != "" {
			ort.SetSharedLibraryPath(libPath)
		}
		envErr = ort.InitializeEnvironment()
	})
	return envErr
}

// Model wraps one loaded Silero VAD v5 ONNX session
type Model struct {
	session *ort.DynamicAdvancedSession
	state   *ort.Tensor[float32]
	sr      *ort.Scalar[int64]
	context [contextSize]float32
}

// loads the Silero VAD v5 ONNX model at modelPath
func NewModel(modelPath, libPath string) (*Model, error) {
	if err := ensureEnvironment(libPath); err != nil {
		return nil, fmt.Errorf("vad: init onnxruntime environment: %w", err)
	}

	opts, err := newSingleThreadedSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("vad: session options: %w", err)
	}
	defer func() { _ = opts.Destroy() }()

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "sr", "state"},
		[]string{"output", "stateN"},
		opts,
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

// newSingleThreadedSessionOptions builds ORT SessionOptions tuned for a tiny
// recurrent model called once every 32ms
func newSingleThreadedSessionOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("new session options: %w", err)
	}
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		_ = opts.Destroy()
		return nil, fmt.Errorf("set intra-op threads: %w", err)
	}
	if err := opts.SetInterOpNumThreads(1); err != nil {
		_ = opts.Destroy()
		return nil, fmt.Errorf("set inter-op threads: %w", err)
	}
	if err := opts.AddSessionConfigEntry("session.intra_op.allow_spinning", "0"); err != nil {
		_ = opts.Destroy()
		return nil, fmt.Errorf("disable intra-op spinning: %w", err)
	}
	if err := opts.AddSessionConfigEntry("session.inter_op.allow_spinning", "0"); err != nil {
		_ = opts.Destroy()
		return nil, fmt.Errorf("disable inter-op spinning: %w", err)
	}
	return opts, nil
}

// Close releases the model's ONNX session and tensors.
func (m *Model) Close() error {
	_ = m.sr.Destroy()
	_ = m.state.Destroy()
	return m.session.Destroy()
}

// Zeroes the recurrent state and trailing context
func (m *Model) Reset() {
	data := m.state.GetData()
	for i := range data {
		data[i] = 0
	}
	m.context = [contextSize]float32{}
}

// runs one inference step over FrameSamples of new mono audio
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
