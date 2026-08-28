---
title: Tracer & Quality Scorer
description: "Record execution traces and score response quality live."
icon: magnifying-glass-chart
---

## Overview

OM1 includes an **execution tracer** that records what the runtime does each turn, and an optional **quality scorer** that evaluates the agent's responses as it runs.

- The **tracer** (`internal/tracer`) writes structured trace events to disk.
- The **quality scorer** subscribes to those trace events and scores each turn for coherence, input classification, and language, exporting the results as Prometheus metrics.

Both are off by default and enabled through the `use_tracer` config block.

## Configuration

```json5
use_tracer: {
  enabled: true,
  quality_scorer: {
    enabled: true,
  },
},
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_tracer.enabled` | bool | `false` | Turns the execution tracer on. |
| `use_tracer.quality_scorer.enabled` | bool | `false` | Turns the live quality scorer on. Requires `use_tracer.enabled: true`. |

> **Note:** If `quality_scorer.enabled` is `true` but `use_tracer.enabled` is `false`, the quality scorer will **not** start (the tracer logs a warning). The scorer depends on the trace event stream.

## Trace output

When enabled, the tracer writes newline-delimited JSON to the `traces/` directory, rotating the file daily:

```
traces/tracer_<YYYY-MM-DD>.jsonl
```

Each line is one trace event from a turn of the core loop.

## Quality scoring

With the quality scorer enabled, each turn is evaluated along three dimensions:

- **Coherence** — is the response coherent with the prompt? (`coherent` / `marginal` / `incoherent`, also mapped to a numeric score of `1` / `0.5` / `0`).
- **Input classification** — how the user input was handled (`positive` / `marginal` / `negative` / `not_addressed`).
- **Language** — the detected spoken language of the turn.

The scores are exported as the `om1_quality_live_*` Prometheus metrics, so you can build a Grafana panel showing response quality trending in real time alongside the latency metrics. See the [Metrics Reference](metrics.md#response-quality-live-quality-scorer) for the exact metric names and types.
