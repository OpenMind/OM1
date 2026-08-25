---
title: Metrics Reference
description: "The Prometheus metrics OM1 exports for observability."
icon: chart-line
---

## Overview

OM1 exposes a Prometheus metrics endpoint so you can monitor the AI pipeline in real time (ASR/LLM/TTS/VLM latencies, knowledge-base queries, response quality, and the internal HTTP proxy).

The metrics server listens on **port `9090`** at `/metrics`. For how to bring up the bundled Prometheus + Grafana stack and the pre-provisioned **OM1 Latency Monitoring** dashboard, see [Quick Start → Prometheus and Grafana Monitoring](1_get-started.md#prometheus-and-grafana-monitoring). This page is the reference for the metrics themselves.

> If you see `Address already in use` on port `9090`, another process (often a local Prometheus) is using it — stop it or free the port.

Most latency metrics come in two forms: a **histogram** (`_seconds`, for quantiles/averages over time) and a **gauge** of the **most recent** value (`_last_seconds`, handy for a live "current latency" panel).

## Pipeline latencies

| Metric | Type | Description |
|--------|------|-------------|
| `om1_llm_latency_seconds` / `_last_seconds` | histogram / gauge | Latency of LLM responses. |
| `om1_vlm_latency_seconds` / `_last_seconds` | histogram / gauge | Latency of VLM (vision-language model) responses. |
| `om1_tts_latency_seconds` / `_last_seconds` | histogram / gauge | Time from a TTS synthesis request to the first audio chunk. |

## ASR & VAD

| Metric | Type | Description |
|--------|------|-------------|
| `om1_asr_latency_seconds` / `_last_seconds` | histogram / gauge | Latency from locally-detected (VAD) end-of-speech to the final transcript. See [VAD & TTS Interrupt](vad_tts_interrupt.md). |
| `om1_asr_speech_duration_seconds` / `_last_seconds` | histogram / gauge | Duration of speech activity from speech-start to speech-end. |
| `om1_asr_utterance_end_latency_seconds` / `_last_seconds` | histogram / gauge | Latency from speech-activity start to end-of-utterance detection. |
| `om1_asr_parallel_transcripts_total` | counter | Transcripts seen by the parallel ASR sensor, labeled by provider model and first-wins outcome. |

## Knowledge base

| Metric | Type | Description |
|--------|------|-------------|
| `om1_kb_query_latency_seconds` / `_last_seconds` | histogram / gauge | Latency of a knowledge base query (embedding + search). |
| `om1_kb_embed_latency_seconds` / `_last_seconds` | histogram / gauge | Latency of the embedding step of a query. |
| `om1_kb_queries_total` | counter | Total knowledge base queries, by outcome. |

## Response quality (live quality scorer)

These are emitted only when the quality scorer is enabled — see [Tracer & Quality Scorer](tracer.md).

| Metric | Type | Description |
|--------|------|-------------|
| `om1_quality_live_active_score` | gauge | Most recent turn's coherence score: coherent = 1, marginal = 0.5, incoherent = 0. |
| `om1_quality_live_turns_scored` | counter | Per-run count of turns that produced a coherence score. |
| `om1_quality_live_coherence_count` | counter | Per-run count of prompt/response pairs by coherence label (coherent / marginal / incoherent). |
| `om1_quality_live_input_classification_count` | counter | Per-run count of user inputs by classification (positive / marginal / negative / not_addressed). |
| `om1_quality_live_language_count` | counter | Per-run count of scored turns by detected spoken language. |

## Internal HTTP proxy

Timings for OM1's proxy to upstream AI services:

| Metric | Type | Description |
|--------|------|-------------|
| `om1_http_proxy_total_seconds` / `_last_seconds` | histogram / gauge | Total proxy time. |
| `om1_http_proxy_parse_seconds` / `_last_seconds` | histogram / gauge | Time the proxy spent parsing the request. |
| `om1_http_upstream_total_seconds` / `_last_seconds` | histogram / gauge | Total upstream service time. |
| `om1_http_upstream_ttfb_seconds` / `_last_seconds` | histogram / gauge | Upstream time-to-first-byte. |

> The authoritative list is defined in `internal/metrics/metrics.go`.
