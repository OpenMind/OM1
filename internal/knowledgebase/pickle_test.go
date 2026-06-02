package knowledgebase

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func pklPath(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	require.NoError(t, err)
	p := filepath.Join(wd, "..", "..", "knowledge_base", "om", "om.pkl")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("pickle data not found at %s, skipping", p)
	}
	return p
}

func TestLoadPickleMetadata(t *testing.T) {
	p := pklPath(t)
	docs, err := loadPickleMetadata(p)
	require.NoError(t, err)
	require.NotEmpty(t, docs)

	for i, doc := range docs {
		require.NotEmpty(t, doc.Text, "doc %d has empty text", i)
		require.NotNil(t, doc.Metadata, "doc %d has nil metadata", i)
	}
}

func TestLoadPickleMetadata_MissingFile(t *testing.T) {
	_, err := loadPickleMetadata("/nonexistent/path.pkl")
	require.Error(t, err)
}

func TestDictHelpers(t *testing.T) {
	// copyMap
	src := map[string]any{"a": 1, "b": "hello"}
	dst := copyMap(src)
	require.Equal(t, src, dst)

	// Mutating dst doesn't affect src.
	dst["a"] = 99
	require.Equal(t, 1, src["a"])
}
