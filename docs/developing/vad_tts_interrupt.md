---
title: VAD & TTS Interrupt
description: "Local voice activity detection, VAD-vs-ASR latency, and barge-in (interrupting the agent while it speaks)."
icon: microphone
---

## Overview

OM1 can run a local **Voice Activity Detection (VAD)** model (Silero VAD v5, via ONNX Runtime) alongside its ASR inputs. The VAD tracker serves two purposes:

1. **VAD-vs-ASR latency measurement** — it records how long it takes from a locally-detected end-of-speech to the final cloud ASR transcript. This drives the `om1_asr_latency_seconds` metric (see [Metrics](metrics.md)).
2. **TTS barge-in (interrupt)** — when enabled, the user can interrupt the agent while it is speaking, instead of waiting for it to finish.

The VAD tracker runs automatically whenever the Silero model and ONNX Runtime are available. If they can't be loaded, VAD is skipped with a log warning and ASR still works normally.

## Prerequisites

VAD is **not** downloaded by `make deps`. Fetch the model and runtime once:

```bash
make download-onnxruntime   # ONNX Runtime shared library (into .onnxruntime/)
make download-vad-model     # Silero VAD v5 model -> models/silero_vad_v5.onnx
```

At runtime OM1 locates the ONNX Runtime shared library automatically (`.onnxruntime/lib/…`, common system paths). You can override it with the `OM1_ONNXRUNTIME_LIB` environment variable.

## Enabling barge-in

Barge-in is configured **per ASR input** via `enable_tts_interrupt`. When it is `false` (the default), incoming audio is ignored while the agent's TTS is playing; when it is `true`, the user's speech is processed even during playback and can interrupt it.

```json5
agent_inputs: [
  {
    type: "GoogleASRInput",
    config: {
      rate: 16000,
      chunk: 1600,
      enable_tts_interrupt: true,   // allow the user to interrupt the agent
    },
  },
]
```

`enable_tts_interrupt` is supported by the ASR inputs (`GoogleASRInput`, `ElevenLabsASRInput`, `RivaASRInput`, their RTSP variants, and `ParallelASRInput`).

## Tuning parameters

These optional fields sit in the same ASR input `config` block:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_tts_interrupt` | bool | `false` | Allow user speech to interrupt the agent's TTS playback (barge-in). |
| `vad_interrupt_confirm_ms` | int | `150` | How long VAD-detected speech must persist before it is treated as a real barge-in (debounces brief noises). |
| `vad_model_path` | string | `models/silero_vad_v5.onnx` | Path to the Silero VAD model. |
| `vad_library_path` | string | auto-resolved | Path to the ONNX Runtime shared library (falls back to `OM1_ONNXRUNTIME_LIB` and standard locations). |
| `vad_output_path` | string | `data/vad_asr_latency.jsonl` | Where VAD-vs-ASR latency records are written (JSONL). |

## Latency records

When the VAD tracker is active it writes one JSONL line per utterance to `vad_output_path`, pairing the locally-detected end-of-speech with the transcript that followed:

```json
{"utterance_ended_at":"...","transcript_at":"...","latency_ms":420.5,"provider":"google","transcript":"give me your paw"}
```

This is useful for measuring perceived responsiveness independently of the cloud ASR provider's own timing.
