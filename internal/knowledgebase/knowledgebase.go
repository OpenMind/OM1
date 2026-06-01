package knowledgebase

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
)

type KnowledgeBase struct {
	retriever *Retriever
	embedder  *HTTPEmbedder
	minScore  float64
	log       *zap.Logger
}

func New(spec *config.KBSpec, log *zap.Logger) (*KnowledgeBase, error) {
	if spec == nil {
		return nil, nil
	}

	name := spec.Name
	if name == "" {
		name = "om"
	}

	baseURL := spec.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:8100"
	}

	cwd, _ := os.Getwd()
	kbDir := filepath.Join(cwd, "knowledge_base", name)
	if _, err := os.Stat(kbDir); err != nil {
		return nil, fmt.Errorf("knowledge base directory not found: %s", kbDir)
	}

	indexPath := filepath.Join(kbDir, name+".faiss")
	metadataPath := filepath.Join(kbDir, name+".pkl")

	retriever, err := NewRetriever(indexPath, metadataPath)
	if err != nil {
		return nil, fmt.Errorf("init retriever: %w", err)
	}

	log.Info("knowledge base initialized",
		zap.Int("documents", retriever.NumDocuments()),
		zap.Int("dimension", retriever.Dimension()),
		zap.String("base_url", baseURL),
		zap.Float64("min_score", spec.MinScore),
	)

	return &KnowledgeBase{
		retriever: retriever,
		embedder:  NewHTTPEmbedder(baseURL),
		minScore:  spec.MinScore,
		log:       log,
	}, nil
}

func (kb *KnowledgeBase) Query(ctx context.Context, question string, topK int) ([]string, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return nil, nil
	}

	if topK <= 0 {
		topK = 3
	}

	embedding, err := kb.embedder.Embed(ctx, question)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	docs, err := kb.retriever.Search(embedding, topK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	if kb.minScore > 0 {
		filtered := docs[:0]
		for _, doc := range docs {
			if doc.Score >= kb.minScore {
				filtered = append(filtered, doc)
			}
		}
		kb.log.Info("knowledge base query",
			zap.Int("query_len", len(question)),
			zap.Int("retrieved", len(docs)),
			zap.Int("after_threshold", len(filtered)),
			zap.Float64("min_score", kb.minScore),
		)
		docs = filtered
	} else {
		kb.log.Info("knowledge base query",
			zap.Int("query_len", len(question)),
			zap.Int("retrieved", len(docs)),
		)
	}

	if len(docs) == 0 {
		return nil, nil
	}

	return formatDocuments(docs, 1500), nil
}

func (kb *KnowledgeBase) Close() {
	if kb.retriever != nil {
		kb.retriever.Close()
	}
}

func formatDocuments(docs []Document, maxChars int) []string {
	var results []string
	totalChars := 0

	for i, doc := range docs {
		source, _ := doc.Metadata["source"].(string)
		if source == "" {
			source = "unknown"
		}
		chunkID := fmt.Sprintf("%v", doc.Metadata["chunk_id"])

		header := fmt.Sprintf("[%d] Source: %s (chunk %s) | Score: %.3f", i+1, source, chunkID, doc.Score)

		content := doc.Text
		if doc.Metadata["type"] == "qa_pair" {
			if ans, ok := doc.Metadata["answer"].(string); ok {
				content = ans
			}
		}

		part := header + "\n" + content
		if totalChars+len(part) > maxChars && len(results) > 0 {
			break
		}

		results = append(results, part)
		totalChars += len(part)
	}

	return results
}
