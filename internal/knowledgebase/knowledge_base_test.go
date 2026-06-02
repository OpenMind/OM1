package knowledgebase

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/config"
)

func TestNew_NilSpec(t *testing.T) {
	kb, err := New(nil, nil)
	require.NoError(t, err)
	require.Nil(t, kb)
}

func TestNew_MissingDir(t *testing.T) {
	spec := &config.KBSpec{Name: "nonexistent", Root: t.TempDir()}
	_, err := New(spec, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "not found")
}

func TestFormatDocuments_Basic(t *testing.T) {
	docs := []Document{
		{Text: "hello world", Metadata: map[string]any{"source": "test.txt", "chunk_id": 0}, Score: 0.95},
		{Text: "second doc", Metadata: map[string]any{"source": "other.txt", "chunk_id": 1}, Score: 0.80},
	}
	results := formatDocuments(docs, 10000)
	require.Len(t, results, 2)
	require.Contains(t, results[0], "test.txt")
	require.Contains(t, results[0], "0.950")
	require.Contains(t, results[0], "hello world")
}

func TestFormatDocuments_QAPair(t *testing.T) {
	docs := []Document{
		{Text: "What is OM1?", Metadata: map[string]any{"type": "qa_pair", "answer": "A robot runtime", "source": "faq", "chunk_id": 0}, Score: 0.9},
	}
	results := formatDocuments(docs, 10000)
	require.Len(t, results, 1)
	require.Contains(t, results[0], "A robot runtime")
	require.NotContains(t, results[0], "What is OM1?")
}

func TestFormatDocuments_TruncatesAtMaxChars(t *testing.T) {
	docs := []Document{
		{Text: "short", Metadata: map[string]any{"source": "a", "chunk_id": 0}, Score: 0.9},
		{Text: "this is a much longer document that should be cut off", Metadata: map[string]any{"source": "b", "chunk_id": 1}, Score: 0.8},
	}
	results := formatDocuments(docs, 50)
	require.Len(t, results, 1)
}

func TestFormatDocuments_Empty(t *testing.T) {
	results := formatDocuments(nil, 1000)
	require.Nil(t, results)
}

func TestFormatDocuments_MissingSource(t *testing.T) {
	docs := []Document{
		{Text: "no source", Metadata: map[string]any{"chunk_id": 0}, Score: 0.5},
	}
	results := formatDocuments(docs, 10000)
	require.Contains(t, results[0], "unknown")
}
