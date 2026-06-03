package knowledgebase

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"
)

func encodeEmbedding(vec []float32) string {
	buf := new(bytes.Buffer)
	for _, f := range vec {
		_ = binary.Write(buf, binary.LittleEndian, math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestNewHTTPEmbedderDefaultBaseURL(t *testing.T) {
	require.Equal(t, DefaultBaseURL, NewHTTPEmbedder("").baseURL)
	require.Equal(t, "http://custom:9", NewHTTPEmbedder("http://custom:9").baseURL)
}

func TestEmbedSuccess(t *testing.T) {
	want := []float32{0.5, -1.25, 3.0}
	var gotBody map[string]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/embed", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_ = json.NewEncoder(w).Encode(embedResponse{EmbeddingB64: encodeEmbedding(want)})
	}))
	t.Cleanup(srv.Close)

	vec, err := NewHTTPEmbedder(srv.URL).Embed(context.Background(), "hello world")
	require.NoError(t, err)
	require.Equal(t, want, vec)
	require.Equal(t, "hello world", gotBody["query"], "query is sent in the request body")
}

func TestEmbedErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("down"))
	}))
	t.Cleanup(srv.Close)

	_, err := NewHTTPEmbedder(srv.URL).Embed(context.Background(), "q")
	require.Error(t, err)
	require.Contains(t, err.Error(), "503")
}

func TestEmbedBadBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(embedResponse{EmbeddingB64: "!!!not base64!!!"})
	}))
	t.Cleanup(srv.Close)

	_, err := NewHTTPEmbedder(srv.URL).Embed(context.Background(), "q")
	require.Error(t, err)
	require.Contains(t, err.Error(), "base64")
}

func TestEmbedNonMultipleOfFour(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// 3 raw bytes is not a whole number of float32s.
		_ = json.NewEncoder(w).Encode(embedResponse{EmbeddingB64: base64.StdEncoding.EncodeToString([]byte{1, 2, 3})})
	}))
	t.Cleanup(srv.Close)

	_, err := NewHTTPEmbedder(srv.URL).Embed(context.Background(), "q")
	require.Error(t, err)
	require.Contains(t, err.Error(), "multiple of 4")
}
