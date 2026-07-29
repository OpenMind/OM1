// Command kbbuild converts RAG-style QA JSON files into the knowledge-base
// format OM1 loads at runtime: an HNSW graph (<name>.graph) plus a metadata
// JSON (<name>.json) of [{id, vector, text, source}] records.
//
// Each question phrasing becomes one indexed record whose vector is the
// embedding of the question (source) and whose text is the answer. Vectors are
// produced by the same embedding service OM1 queries against (default
// http://localhost:8100/embed, e5-small-v2), so build-time and query-time
// embeddings are identical.
//
// Usage:
//
//	go run ./cmd/kbbuild -name trmi -out ./knowledge_base trmi_qa.json om_qa.json
//
// Input QA JSON supports both formats (mixed OK):
//
//	Format A: {"questions": ["...", "..."], "a": "answer"}
//	Format B: {"q": "...", "a": "answer"}   (also accepts "question"/"answer")
package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/coder/hnsw"
)

// record is one metadata entry, matching the shape OM1's knowledgebase loader
// unmarshals (internal/knowledgebase: document{id, vector, text, source}).
type record struct {
	ID     int       `json:"id"`
	Vector []float32 `json:"vector"`
	Text   string    `json:"text"`
	Source string    `json:"source"`
}

// qaEntry accepts both the grouped (Format A) and flat (Format B) shapes.
type qaEntry struct {
	Questions []string `json:"questions"`
	Q         string   `json:"q"`
	Question  string   `json:"question"`
	A         string   `json:"a"`
	Answer    string   `json:"answer"`
}

func main() {
	var (
		name     = flag.String("name", "", "knowledge base name (e.g. trmi); outputs <out>/<name>/<name>.{graph,json}")
		outRoot  = flag.String("out", "./knowledge_base", "knowledge base root directory")
		embedURL = flag.String("embed-url", "http://localhost:8100", "embedding service base URL (POST /embed)")
	)
	flag.Parse()

	if *name == "" {
		fmt.Fprintln(os.Stderr, "error: -name is required")
		flag.Usage()
		os.Exit(1)
	}
	files := flag.Args()
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "error: provide at least one QA JSON file")
		os.Exit(1)
	}

	questions, answers, err := loadQA(files)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load QA: %v\n", err)
		os.Exit(1)
	}
	if len(questions) == 0 {
		fmt.Fprintln(os.Stderr, "error: no QA pairs found")
		os.Exit(1)
	}
	fmt.Printf("Loaded %d question phrasings from %d file(s)\n", len(questions), len(files))

	client := &http.Client{Timeout: 30 * time.Second}
	graph := hnsw.NewGraph[int]()
	records := make([]record, 0, len(questions))

	for i, q := range questions {
		vec, err := embed(client, *embedURL, q)
		if err != nil {
			fmt.Fprintf(os.Stderr, "embed question %d (%q): %v\n", i, q, err)
			os.Exit(1)
		}
		graph.Add(hnsw.MakeNode(i, vec))
		records = append(records, record{ID: i, Vector: vec, Text: answers[i], Source: q})
		if (i+1)%25 == 0 || i+1 == len(questions) {
			fmt.Printf("  embedded %d/%d\n", i+1, len(questions))
		}
	}

	kbDir := filepath.Join(*outRoot, *name)
	if err := os.MkdirAll(kbDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir %s: %v\n", kbDir, err)
		os.Exit(1)
	}

	graphPath := filepath.Join(kbDir, *name+".graph")
	if err := writeGraph(graphPath, graph); err != nil {
		fmt.Fprintf(os.Stderr, "write graph: %v\n", err)
		os.Exit(1)
	}

	metaPath := filepath.Join(kbDir, *name+".json")
	if err := writeJSON(metaPath, records); err != nil {
		fmt.Fprintf(os.Stderr, "write metadata: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Done. Wrote %s (%d vectors) and %s (%d records)\n",
		graphPath, graph.Len(), metaPath, len(records))
}

// loadQA reads QA files and returns parallel (questions, answers) slices, one
// entry per question phrasing (Format A expands each phrasing to the answer).
func loadQA(files []string) (questions, answers []string, err error) {
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", f, err)
		}
		var entries []qaEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, nil, fmt.Errorf("parse %s: %w", f, err)
		}
		count := 0
		for _, e := range entries {
			a := e.A
			if a == "" {
				a = e.Answer
			}
			phrasings := e.Questions
			if len(phrasings) == 0 {
				q := e.Q
				if q == "" {
					q = e.Question
				}
				if q != "" {
					phrasings = []string{q}
				}
			}
			for _, q := range phrasings {
				questions = append(questions, q)
				answers = append(answers, a)
				count++
			}
		}
		fmt.Printf("  %s: %d pairs\n", f, count)
	}
	return questions, answers, nil
}

// embed calls POST <baseURL>/embed with {"query": text} and decodes the
// base64 little-endian float32 vector, matching internal/knowledgebase.
func embed(client *http.Client, baseURL, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]string{"query": text})
	resp, err := client.Post(baseURL+"/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		return nil, fmt.Errorf("embed service returned %s: %s", resp.Status, snippet)
	}
	var decoded struct {
		EmbeddingB64 string `json:"embedding_b64"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	rawVec, err := base64.StdEncoding.DecodeString(decoded.EmbeddingB64)
	if err != nil {
		return nil, fmt.Errorf("base64 decode: %w", err)
	}
	if len(rawVec)%4 != 0 {
		return nil, fmt.Errorf("embedding byte length %d not a multiple of 4", len(rawVec))
	}
	vec := make([]float32, len(rawVec)/4)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(rawVec[i*4 : i*4+4]))
	}
	return vec, nil
}

func writeGraph(path string, graph *hnsw.Graph[int]) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	if err := graph.Export(f); err != nil {
		return err
	}
	return nil
}

func writeJSON(path string, records []record) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(records)
}
