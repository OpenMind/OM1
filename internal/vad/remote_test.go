package vad

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewRemoteModelDefaultBaseURL(t *testing.T) {
	require.Equal(t, DefaultServiceURL, NewRemoteModel("").baseURL)
	require.Equal(t, "http://custom:9", NewRemoteModel("http://custom:9").baseURL)
}

func TestRemoteModelInferSendsFramedInputAndCarriesState(t *testing.T) {
	var requests []inferRequest
	nextState := make([]float32, 2*stateDim)
	for i := range nextState {
		nextState[i] = float32(i)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/infer", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req inferRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		requests = append(requests, req)

		require.Equal(t, SampleRate, req.SR)
		_ = json.NewEncoder(w).Encode(inferResponse{
			Prob:     0.73,
			StateB64: encodeFloat32s(nextState),
		})
	}))
	t.Cleanup(srv.Close)

	m := NewRemoteModel(srv.URL)

	frame1 := make([]float32, FrameSamples)
	for i := range frame1 {
		frame1[i] = 0.1
	}
	prob, err := m.Infer(frame1)
	require.NoError(t, err)
	require.Equal(t, float32(0.73), prob)
	require.Len(t, requests, 1)

	sentInput1, err := decodeFloat32s(requests[0].InputB64, modelInputSamples)
	require.NoError(t, err)
	// First call: no prior context yet, so the leading contextSize samples are zero.
	require.Equal(t, make([]float32, contextSize), sentInput1[:contextSize])
	require.Equal(t, frame1, sentInput1[contextSize:])

	sentState1, err := decodeFloat32s(requests[0].StateB64, 2*stateDim)
	require.NoError(t, err)
	require.Equal(t, make([]float32, 2*stateDim), sentState1, "first call carries zero state")

	frame2 := make([]float32, FrameSamples)
	for i := range frame2 {
		frame2[i] = 0.2
	}
	_, err = m.Infer(frame2)
	require.NoError(t, err)
	require.Len(t, requests, 2)

	sentInput2, err := decodeFloat32s(requests[1].InputB64, modelInputSamples)
	require.NoError(t, err)
	require.Equal(t, frame1[FrameSamples-contextSize:], sentInput2[:contextSize],
		"second call's leading context is the tail of the first frame")
	require.Equal(t, frame2, sentInput2[contextSize:])

	sentState2, err := decodeFloat32s(requests[1].StateB64, 2*stateDim)
	require.NoError(t, err)
	require.Equal(t, nextState, sentState2, "second call carries the state returned by the first")
}

func TestRemoteModelInferWrongFrameSize(t *testing.T) {
	m := NewRemoteModel("http://unused")
	_, err := m.Infer(make([]float32, FrameSamples-1))
	require.Error(t, err)
	require.Contains(t, err.Error(), "512")
}

func TestRemoteModelInferErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("down"))
	}))
	t.Cleanup(srv.Close)

	m := NewRemoteModel(srv.URL)
	_, err := m.Infer(make([]float32, FrameSamples))
	require.Error(t, err)
	require.Contains(t, err.Error(), "503")
}

func TestRemoteModelResetZeroesStateAndContext(t *testing.T) {
	var lastReq inferRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&lastReq)
		_ = json.NewEncoder(w).Encode(inferResponse{
			Prob:     0.1,
			StateB64: encodeFloat32s(make([]float32, 2*stateDim)),
		})
	}))
	t.Cleanup(srv.Close)

	m := NewRemoteModel(srv.URL)
	frame := make([]float32, FrameSamples)
	for i := range frame {
		frame[i] = 0.5
	}
	_, err := m.Infer(frame)
	require.NoError(t, err)

	m.Reset()
	require.Equal(t, [2 * stateDim]float32{}, m.state)
	require.Equal(t, [contextSize]float32{}, m.context)

	_, err = m.Infer(frame)
	require.NoError(t, err)
	sentInput, err := decodeFloat32s(lastReq.InputB64, modelInputSamples)
	require.NoError(t, err)
	require.Equal(t, make([]float32, contextSize), sentInput[:contextSize],
		"after Reset, leading context is zero again")
}

func TestRemoteModelClose(t *testing.T) {
	require.NoError(t, NewRemoteModel("http://unused").Close())
}
