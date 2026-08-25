---
title: Observability
description: "Monitor the OM1 runtime with metrics, traces, and quality scoring."
icon: gauge-high
---

## Overview

OM1 exposes tooling to observe the runtime in real time — from low-level pipeline latencies to per-turn response quality.

This section covers:

- [**Metrics Reference**](metrics.md) — the Prometheus metrics OM1 exports (ASR/LLM/TTS/VLM latencies, knowledge-base queries, response quality, and the internal HTTP proxy).
- [**Tracer & Quality Scorer**](tracer.md) — record structured execution traces each turn and score response quality live.
