package knowledgebase

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBytesToFloat32_Valid(t *testing.T) {
	floats := []float32{1.0, -2.5, 3.14}
	raw := float32sToBytes(floats)

	got, err := bytesToFloat32(raw)
	require.NoError(t, err)
	require.Len(t, got, 3)
	require.InDelta(t, 1.0, got[0], 1e-6)
	require.InDelta(t, -2.5, got[1], 1e-6)
	require.InDelta(t, 3.14, got[2], 1e-4)
}

func TestBytesToFloat32_NotDivisibleBy4(t *testing.T) {
	_, err := bytesToFloat32([]byte{1, 2, 3})
	require.Error(t, err)
	require.Contains(t, err.Error(), "not divisible by 4")
}

func TestBytesToFloat32_Empty(t *testing.T) {
	got, err := bytesToFloat32([]byte{})
	require.NoError(t, err)
	require.Empty(t, got)
}

func TestHTTPEmbedder_Embed(t *testing.T) {
	expected := []float32{0.1, 0.2, 0.3}
	b64 := base64.StdEncoding.EncodeToString(float32sToBytes(expected))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embed", r.URL.Path)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"embedding_b64":"` + b64 + `","latency_ms":1.5}`))
	}))
	defer srv.Close()

	embedder := NewHTTPEmbedder(srv.URL)
	got, err := embedder.Embed(context.Background(), "hello")
	require.NoError(t, err)
	require.Len(t, got, 3)
	require.InDelta(t, 0.1, got[0], 1e-6)
	require.InDelta(t, 0.2, got[1], 1e-6)
	require.InDelta(t, 0.3, got[2], 1e-6)
}

func TestHTTPEmbedder_EmbedBatch(t *testing.T) {
	v1 := []float32{1.0, 2.0}
	v2 := []float32{3.0, 4.0}
	b1 := base64.StdEncoding.EncodeToString(float32sToBytes(v1))
	b2 := base64.StdEncoding.EncodeToString(float32sToBytes(v2))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embed_batch", r.URL.Path)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"embeddings_b64":["` + b1 + `","` + b2 + `"],"latency_ms":2.0}`))
	}))
	defer srv.Close()

	embedder := NewHTTPEmbedder(srv.URL)
	got, err := embedder.EmbedBatch(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, got, 2)
	require.InDelta(t, 1.0, got[0][0], 1e-6)
	require.InDelta(t, 4.0, got[1][1], 1e-6)
}

func TestHTTPEmbedder_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	embedder := NewHTTPEmbedder(srv.URL)
	_, err := embedder.Embed(context.Background(), "fail")
	require.Error(t, err)
	require.Contains(t, err.Error(), "500")
}

// float32sToBytes converts a slice of float32 to little-endian bytes.
func float32sToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}
