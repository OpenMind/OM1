package knowledgebase

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/types"
)

// loadPickleMetadata loads document metadata from a Python pickle file.
func loadPickleMetadata(path string) ([]Document, error) {
	raw, err := pickle.Load(path)
	if err != nil {
		return nil, fmt.Errorf("load pickle: %w", err)
	}

	switch v := raw.(type) {
	case *types.List:
		return parseListMetadata(v)
	case *types.Dict:
		return parseDictMetadata(v)
	default:
		return nil, fmt.Errorf("unsupported pickle type: %T", raw)
	}
}

// parseListMetadata handles the list-of-dicts format.
func parseListMetadata(list *types.List) ([]Document, error) {
	docs := make([]Document, 0, list.Len())
	for i := 0; i < list.Len(); i++ {
		item := list.Get(i)
		d, ok := item.(*types.Dict)
		if !ok {
			return nil, fmt.Errorf("list item %d: expected dict, got %T", i, item)
		}

		text := dictGetString(d, "text")
		meta := dictToMap(dictGetDict(d, "metadata"))

		docs = append(docs, Document{Text: text, Metadata: meta})
	}
	return docs, nil
}

// parseDictMetadata handles the QA pair format.
func parseDictMetadata(d *types.Dict) ([]Document, error) {
	questionsRaw := dictGet(d, "questions")
	answersRaw := dictGet(d, "answers")

	if questionsRaw == nil || answersRaw == nil {
		return nil, fmt.Errorf("dict must have 'questions' and 'answers' keys, got: %v", dictKeys(d))
	}

	questions, ok := questionsRaw.(*types.List)
	if !ok {
		return nil, fmt.Errorf("questions: expected list, got %T", questionsRaw)
	}
	answers, ok := answersRaw.(*types.List)
	if !ok {
		return nil, fmt.Errorf("answers: expected list, got %T", answersRaw)
	}

	if questions.Len() != answers.Len() {
		return nil, fmt.Errorf("questions (%d) and answers (%d) length mismatch", questions.Len(), answers.Len())
	}

	docs := make([]Document, questions.Len())
	for i := 0; i < questions.Len(); i++ {
		q := fmt.Sprintf("%v", questions.Get(i))
		a := fmt.Sprintf("%v", answers.Get(i))
		docs[i] = Document{
			Text: q,
			Metadata: map[string]any{
				"answer":   a,
				"type":     "qa_pair",
				"chunk_id": i,
			},
		}
	}
	return docs, nil
}

func dictGet(d *types.Dict, key string) any {
	for _, entry := range *d {
		if k, ok := entry.Key.(string); ok && k == key {
			return entry.Value
		}
	}
	return nil
}

func dictGetString(d *types.Dict, key string) string {
	v := dictGet(d, key)
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func dictGetDict(d *types.Dict, key string) *types.Dict {
	v := dictGet(d, key)
	if v == nil {
		return nil
	}
	if dd, ok := v.(*types.Dict); ok {
		return dd
	}
	return nil
}

func dictToMap(d *types.Dict) map[string]any {
	if d == nil {
		return map[string]any{}
	}
	m := make(map[string]any, len(*d))
	for _, entry := range *d {
		if k, ok := entry.Key.(string); ok {
			m[k] = entry.Value
		}
	}
	return m
}

func dictKeys(d *types.Dict) []string {
	keys := make([]string, 0, len(*d))
	for _, entry := range *d {
		if k, ok := entry.Key.(string); ok {
			keys = append(keys, k)
		}
	}
	return keys
}
