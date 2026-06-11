package memory

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBM25Index_AddAndSearch(t *testing.T) {
	idx := NewBM25Index()

	idx.Add("h1", MemoryEntry{Text: "Hello my name is David"})
	idx.Add("h2", MemoryEntry{Text: "The weather is sunny today"})
	idx.Add("h3", MemoryEntry{Text: "David likes playing basketball"})

	require.Equal(t, 3, idx.Size())

	results := idx.Search("David", 5, "")
	require.GreaterOrEqual(t, len(results), 2, "should match both docs containing David")
	// Top result should mention David.
	require.Contains(t, results[0].Text, "David")
}

func TestBM25Index_DuplicateAdd(t *testing.T) {
	idx := NewBM25Index()

	idx.Add("h1", MemoryEntry{Text: "Hello world"})
	idx.Add("h1", MemoryEntry{Text: "Different text same hash"})

	require.Equal(t, 1, idx.Size(), "duplicate hash should be no-op")
	results := idx.Search("Hello", 5, "")
	require.Len(t, results, 1)
	require.Equal(t, "Hello world", results[0].Text)
}

func TestBM25Index_EmptyQuery(t *testing.T) {
	idx := NewBM25Index()
	idx.Add("h1", MemoryEntry{Text: "Hello world"})

	results := idx.Search("", 5, "")
	require.Nil(t, results)
}

func TestBM25Index_EmptyIndex(t *testing.T) {
	idx := NewBM25Index()

	results := idx.Search("anything", 5, "")
	require.Nil(t, results)
}

func TestBM25Index_TopKLimit(t *testing.T) {
	idx := NewBM25Index()
	for i := 0; i < 10; i++ {
		idx.Add(hashText("doc"+string(rune('a'+i))), MemoryEntry{Text: "common term unique" + string(rune('a'+i))})
	}

	results := idx.Search("common", 3, "")
	require.LessOrEqual(t, len(results), 3)
}

func TestBM25Index_IDFRanking(t *testing.T) {
	idx := NewBM25Index()

	// "hello" appears in both docs, "basketball" only in one.
	idx.Add("h1", MemoryEntry{Text: "hello world hello"})
	idx.Add("h2", MemoryEntry{Text: "hello basketball game"})

	results := idx.Search("basketball", 5, "")
	require.Len(t, results, 1)
	require.Contains(t, results[0].Text, "basketball")
}

func TestTokenize(t *testing.T) {
	tokens := tokenize("Hello World  TEST")
	require.Equal(t, []string{"hello", "world", "test"}, tokens)
}
