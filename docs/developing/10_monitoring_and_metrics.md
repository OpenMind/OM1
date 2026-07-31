---
title: Monitoring and Metrics
description: "Understanding OM1's Prometheus metrics for monitoring and observability."
icon: chart-line
---

# Monitoring and Metrics

OM1 exposes a variety of metrics in the [Prometheus format](https://prometheus.io/docs/instrumenting/exposition_formats/), allowing you to monitor the health, performance, and behavior of your agent. These metrics are available via an HTTP endpoint that can be scraped by a Prometheus server and visualized in tools like Grafana.

## Quality Scorer Metrics

When the `quality_scorer` is enabled via the [`use_tracer` config](./3_configuration.md#tracer-configuration), OM1 uses a live LLM to analyze conversation quality. It generates the following metrics, which are invaluable for understanding user engagement and interaction quality.

A Grafana dashboard for these metrics is available in the repository at `/grafana/dashboards/om1-quality-dashboard.json`.

| Metric Name                                 | Type      | Description                                                                                             | Labels                                  |
| ------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `om1_quality_live_active_score`             | Gauge     | The most recent turn's coherence score. `coherent`=1.0, `marginal`=0.5, `incoherent`=0.0.                 |                                         |
| `om1_quality_live_coherence_count`          | Counter   | A per-run count of prompt/response pairs by their classified coherence.                                 | `coherence` (coherent, marginal, incoherent) |
| `om1_quality_live_input_classification_count` | Counter   | A per-run count of user inputs by their classified intent and tone.                                     | `label` (positive, marginal, negative, not_addressed) |
| `om1_quality_live_language_count`           | Counter   | A per-run count of scored turns, by detected spoken language.                                           | `language` (e.g., English, Spanish)     |
| `om1_quality_live_turns_scored`             | Counter   | The total number of conversation turns that have been successfully scored for coherence.                |                                         |

### Understanding the Labels

*   **Coherence:**
    *   `coherent`: The robot's response directly and relevantly addresses the user's prompt.
    *   `marginal`: The robot understood the prompt but deflected or gave an unrelated response.
    *   `incoherent`: The robot misunderstood or ignored the prompt entirely.
*   **Input Classification:**
    *   `positive`: A genuine, engaged question or remark directed at the robot.
    *   `marginal`: Mild dissatisfaction or a brush-off, but not a direct complaint.
    *   `negative`: Direct frustration or a complaint that the robot is malfunctioning.
    *   `not_addressed`: Ambient chatter not directed at the robot.

## Other Core Metrics

OM1 also exposes core performance metrics that are always available.

| Metric Name             | Type      | Description                                         | Labels |
| ----------------------- | --------- | --------------------------------------------------- | ------ |
| `om1_llm_latency_ms`    | Histogram | End-to-end latency for LLM requests, in milliseconds. |        |
| `om1_kb_queries_total`  | Counter   | Total number of knowledge base queries.             | `status` (ok, error) |

This list is not exhaustive. You can discover all available metrics by inspecting the Prometheus endpoint.
