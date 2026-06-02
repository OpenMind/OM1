package knowledgebase

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func kbRoot(t *testing.T) string {
	t.Helper()
	// Walk up from the test file to find the repo root.
	wd, err := os.Getwd()
	require.NoError(t, err)
	root := filepath.Join(wd, "..", "..", "knowledge_base", "om")
	if _, err := os.Stat(filepath.Join(root, "om.faiss")); err != nil {
		t.Skipf("knowledge base data not found at %s, skipping", root)
	}
	return root
}

func TestNewRetriever(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)
	defer r.Close()

	require.Greater(t, r.NumDocuments(), 0)
	require.Greater(t, r.Dimension(), 0)
}

func TestRetriever_Search(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)
	defer r.Close()

	// Create a zero vector with the correct dimension.
	query := make([]float32, r.Dimension())
	docs, err := r.Search(query, 3)
	require.NoError(t, err)
	require.NotEmpty(t, docs)
	require.LessOrEqual(t, len(docs), 3)

	for _, doc := range docs {
		require.NotEmpty(t, doc.Text)
		require.NotNil(t, doc.Metadata)
	}
}

func TestRetriever_SearchWrongDimension(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)
	defer r.Close()

	query := make([]float32, r.Dimension()+1)
	_, err = r.Search(query, 3)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dim=")
}

func TestRetriever_BatchSearch(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)
	defer r.Close()

	q1 := make([]float32, r.Dimension())
	q2 := make([]float32, r.Dimension())
	q2[0] = 1.0

	results, err := r.BatchSearch([][]float32{q1, q2}, 2)
	require.NoError(t, err)
	require.Len(t, results, 2)

	for _, docs := range results {
		require.NotEmpty(t, docs)
		require.LessOrEqual(t, len(docs), 2)
	}
}

func TestRetriever_BatchSearchEmpty(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)
	defer r.Close()

	results, err := r.BatchSearch(nil, 3)
	require.NoError(t, err)
	require.Nil(t, results)
}

func TestRetriever_MissingIndex(t *testing.T) {
	_, err := NewRetriever("/nonexistent/path.faiss", "/nonexistent/path.pkl")
	require.Error(t, err)
	require.Contains(t, err.Error(), "not found")
}

func TestRetriever_CloseIdempotent(t *testing.T) {
	root := kbRoot(t)
	r, err := NewRetriever(filepath.Join(root, "om.faiss"), filepath.Join(root, "om.pkl"))
	require.NoError(t, err)

	r.Close()
	r.Close() // should not panic
}
