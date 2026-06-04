package knowledgebase

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/coder/hnsw"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/metrics"
)

// defaultTopK is the default number of results to return for a query if not specified in the KB spec.
const defaultTopK = 3

// document is one indexed record loaded from the metadata JSON.
type document struct {
	ID     int       `json:"id"`
	Vector []float32 `json:"vector"`
	Text   string    `json:"text"`
	Source string    `json:"source"`
}

// KnowledgeBase retrieves documents relevant to a query using an HNSW index.
type KnowledgeBase struct {
	graph    *hnsw.Graph[int]
	docs     map[int]document
	embedder Embedder
	topK     int
	minScore float64
}

// NewKnowledgeBase constructs a KnowledgeBase from the given spec.
func NewKnowledgeBase(spec *config.KBSpec) (*KnowledgeBase, error) {
	if spec == nil {
		return nil, fmt.Errorf("knowledge base spec is nil")
	}
	if spec.Name == "" {
		return nil, fmt.Errorf("knowledge_base_name is required")
	}

	kbDir, err := resolveKBDir(spec.Root, spec.Name)
	if err != nil {
		return nil, err
	}

	graphPath := filepath.Join(kbDir, spec.Name+".graph")
	metadataPath := filepath.Join(kbDir, spec.Name+".json")

	graph, err := loadGraph(graphPath)
	if err != nil {
		return nil, err
	}

	docs, err := loadDocuments(metadataPath)
	if err != nil {
		return nil, err
	}

	if graph.Len() != len(docs) {
		fmt.Fprintf(os.Stderr, "knowledgebase: index has %d vectors but metadata has %d documents\n", graph.Len(), len(docs))
	}

	topK := spec.TopK
	if topK <= 0 {
		topK = defaultTopK
	}

	return &KnowledgeBase{
		graph:    graph,
		docs:     docs,
		embedder: NewHTTPEmbedder(spec.BaseURL),
		topK:     topK,
		minScore: spec.MinScore,
	}, nil
}

// Query embeds the question, searches the graph for the nearest documents.
func (kb *KnowledgeBase) Query(ctx context.Context, question string, topK int) (results []string, err error) {
	start := time.Now()
	embedSeconds := -1.0
	defer func() {
		metrics.RecordKBQuery(embedSeconds, time.Since(start).Seconds(), err == nil)
	}()

	if topK <= 0 {
		topK = kb.topK
	}

	embedStart := time.Now()
	queryVec, err := kb.embedder.Embed(ctx, question)
	embedSeconds = time.Since(embedStart).Seconds()
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	searchK := topK * 5
	if searchK > kb.graph.Len() {
		searchK = kb.graph.Len()
	}
	if searchK == 0 {
		return nil, nil
	}

	nodes := kb.graph.Search(queryVec, searchK)

	results = make([]string, 0, topK)
	seen := make(map[string]struct{}, topK)
	for _, node := range nodes {
		doc, ok := kb.docs[node.Key]
		if !ok {
			continue
		}
		if _, dup := seen[doc.Text]; dup {
			continue
		}

		if kb.minScore > 0 {
			similarity := 1 - float64(kb.graph.Distance(queryVec, node.Value))
			if similarity < kb.minScore {
				continue
			}
		}

		seen[doc.Text] = struct{}{}
		results = append(results, doc.Text)
		if len(results) >= topK {
			break
		}
	}

	return results, nil
}

// loadGraph reads a coder/hnsw index previously serialized via Graph.Export.
func loadGraph(path string) (*hnsw.Graph[int], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open index %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	graph := hnsw.NewGraph[int]()
	if err := graph.Import(bufio.NewReader(f)); err != nil {
		return nil, fmt.Errorf("import index %s: %w", path, err)
	}
	return graph, nil
}

// loadDocuments reads the metadata JSON (a list of records) into a map keyed by id.
func loadDocuments(path string) (map[int]document, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read metadata %s: %w", path, err)
	}

	var records []document
	if err := json.Unmarshal(raw, &records); err != nil {
		return nil, fmt.Errorf("parse metadata %s: %w", path, err)
	}

	docs := make(map[int]document, len(records))
	for _, rec := range records {
		docs[rec.ID] = rec
	}
	return docs, nil
}

// resolveKBDir locates the knowledge base directory "<root>/<name>". When an
// explicit root is configured it is used directly; otherwise several default
// roots are tried so the lookup works whether the binary is run from the build
// directory or via `go run` (where os.Executable points at a temp path):
//
//   - <cwd>/knowledge_base                (running from the repo root)
//   - <exe dir>/../../knowledge_base      (the built ./build/om1 binary)
//   - <exe dir>/knowledge_base
//
// It returns the first candidate that exists, or an error listing what was tried.
func resolveKBDir(root, name string) (string, error) {
	var roots []string
	if root != "" {
		roots = []string{root}
	} else {
		if cwd, err := os.Getwd(); err == nil {
			roots = append(roots, filepath.Join(cwd, "knowledge_base"))
		}
		if exe, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exe)
			roots = append(roots,
				filepath.Join(exeDir, "..", "..", "knowledge_base"),
				filepath.Join(exeDir, "knowledge_base"),
			)
		}
	}

	tried := make([]string, 0, len(roots))
	for _, r := range roots {
		kbDir := filepath.Join(r, name)
		tried = append(tried, kbDir)
		if info, err := os.Stat(kbDir); err == nil && info.IsDir() {
			return kbDir, nil
		}
	}
	return "", fmt.Errorf("knowledge base %q not found; tried: %v", name, tried)
}
