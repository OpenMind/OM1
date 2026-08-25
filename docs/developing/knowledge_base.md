---
title: Knowledge Base (RAG)
description: "Give your agent retrieval-augmented context from a local vector index."
icon: book
---

## Overview

OM1 supports **retrieval-augmented generation (RAG)**: before each LLM call, the runtime can retrieve the most relevant documents for the current context and include them in the prompt.

The knowledge base has two parts:

- A **local vector index** — an [HNSW](https://github.com/coder/hnsw) graph plus a metadata JSON file, loaded from a knowledge-base directory on disk (`internal/knowledgebase`).
- An **embedding service** — an external HTTP endpoint that turns a query string into an embedding vector so it can be matched against the index.

## Configuration

Add a `knowledge_base` block to your config:

```json5
knowledge_base: {
  knowledge_base_name: "om",                        // required — the KB directory/file name
  base_url: "${KB_BASE_URL:-http://localhost:8100}", // embedding service endpoint
  min_score: 0.6,                                    // drop results below this similarity
  top_k: 3,                                          // number of documents to retrieve
},
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `knowledge_base_name` | string | Yes | Name of the knowledge base. The runtime loads `<name>.graph` and `<name>.json` from the KB directory. |
| `base_url` | string | No | Embedding service URL. Defaults to `http://localhost:8100`. Any trailing slash is automatically removed. |
| `knowledge_base_root` | string | No | Explicit root directory for knowledge bases. If omitted, OM1 auto-resolves it (see below). |
| `top_k` | int | No | Number of documents to return per query. Default `3`. |
| `min_score` | float | No | Minimum similarity score; lower-scoring matches are discarded. |

## Where the index is loaded from

If `knowledge_base_root` is not set, OM1 looks for a `knowledge_base/<name>/` directory in these locations, in order:

1. `<cwd>/knowledge_base` — when running from the repo root
2. `<exe dir>/../../knowledge_base` — for the built `./build/om1` binary
3. `<exe dir>/knowledge_base`

The repository ships an example KB at [`knowledge_base/om/`](https://github.com/OpenMind/OM1/tree/main/knowledge_base/om) containing `om.graph` (the HNSW index) and `om.json` (the document metadata: id, vector, text, source).

## The embedding service

A query is embedded by POSTing it to the service at `base_url`. Run your embedding service (or point `KB_BASE_URL` at a hosted one) before starting an agent that uses a knowledge base. If the embedding step fails, the query is skipped and the failure is reflected in the `om1_kb_queries_total` metric.

## Observability

Knowledge base activity is exported to Prometheus (see [Metrics](metrics.md)):

- `om1_kb_query_latency_seconds` — full query latency (embedding + search)
- `om1_kb_embed_latency_seconds` — embedding-step latency
- `om1_kb_queries_total` — total queries, labeled by outcome
