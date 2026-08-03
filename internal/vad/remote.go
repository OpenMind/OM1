package vad

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/openmind/om1/internal/httpclient"
)

// DefaultServiceURL is the base URL of the VAD GPU inference service
// (see docker/Dockerfile.vad in OM1-modules) used when none is configured.
const DefaultServiceURL = "http://localhost:8300"

const inferTimeout = 2 * time.Second

// RemoteModel runs Silero VAD v5 inference through a remote GPU
// microservice instead of loading the ONNX graph locally, moving the
// per-frame inference cost off the CPU. It reproduces the same
// context/state bookkeeping as Model, so it satisfies the same Inferer
// interface Segmenter expects and is a drop-in replacement for it.
type RemoteModel struct {
	baseURL string
	client  *http.Client

	state   [2 * stateDim]float32
	context [contextSize]float32
}

// NewRemoteModel builds a RemoteModel that calls the VAD GPU service at
// baseURL (DefaultServiceURL if empty).
func NewRemoteModel(baseURL string) *RemoteModel {
	if baseURL == "" {
		baseURL = DefaultServiceURL
	}
	return &RemoteModel{baseURL: baseURL, client: httpclient.Default()}
}

// inferRequest is the JSON body sent to the VAD service's /infer endpoint.
type inferRequest struct {
	InputB64 string `json:"input_b64"`
	StateB64 string `json:"state_b64"`
	SR       int    `json:"sr"`
}

// inferResponse is the JSON shape returned by the /infer endpoint.
type inferResponse struct {
	Prob     float32 `json:"prob"`
	StateB64 string  `json:"state_b64"`
}

// Infer runs one inference step over FrameSamples of new mono audio by
// calling the remote VAD service, matching Model.Infer's contract exactly:
// same input framing (trailing context + new frame), same recurrent state
// carried between calls.
func (r *RemoteModel) Infer(frame []float32) (float32, error) {
	if len(frame) != FrameSamples {
		return 0, fmt.Errorf("vad: frame must be %d samples, got %d", FrameSamples, len(frame))
	}

	var input [modelInputSamples]float32
	copy(input[:contextSize], r.context[:])
	copy(input[contextSize:], frame)

	reqBody, err := json.Marshal(inferRequest{
		InputB64: encodeFloat32s(input[:]),
		StateB64: encodeFloat32s(r.state[:]),
		SR:       SampleRate,
	})
	if err != nil {
		return 0, fmt.Errorf("vad: marshal infer request: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), inferTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.baseURL+"/infer", bytes.NewReader(reqBody))
	if err != nil {
		return 0, fmt.Errorf("vad: build infer request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("vad: infer request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, fmt.Errorf("vad: service returned %s: %s", resp.Status, snippet)
	}

	var decoded inferResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return 0, fmt.Errorf("vad: decode infer response: %w", err)
	}

	newState, err := decodeFloat32s(decoded.StateB64, 2*stateDim)
	if err != nil {
		return 0, fmt.Errorf("vad: decode state: %w", err)
	}
	copy(r.state[:], newState)
	copy(r.context[:], input[modelInputSamples-contextSize:])

	return decoded.Prob, nil
}

// Reset zeroes the recurrent state and trailing context.
func (r *RemoteModel) Reset() {
	r.state = [2 * stateDim]float32{}
	r.context = [contextSize]float32{}
}

// Close is a no-op: RemoteModel holds no local resources. Its HTTP client
// is the shared, process-wide client from internal/httpclient, which
// outlives any single RemoteModel.
func (r *RemoteModel) Close() error { return nil }

func encodeFloat32s(v []float32) string {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func decodeFloat32s(b64 string, expectedLen int) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return nil, fmt.Errorf("base64 decode: %w", err)
	}
	if len(raw) != expectedLen*4 {
		return nil, fmt.Errorf("expected %d bytes, got %d", expectedLen*4, len(raw))
	}
	out := make([]float32, expectedLen)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}
