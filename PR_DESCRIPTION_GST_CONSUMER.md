# Consume the synchronized A+V stream + stamp ASR audio at capture

Pairs OM1 with the OM1-video-processor GStreamer pipeline, which serves a single
muxed A+V RTSP session (`rtsp://localhost:8555/live`) on one clock. Two parts:
config (repoint consumers) and code (stamp audio capture time at ingest).

## Config — repoint consumers (non-breaking)

`config/unitree_g1_conversation.json5` is env-parametrized; **defaults preserve
current behavior exactly**:

- ASR: `type: "${ASR_INPUT_PLUGIN:-GoogleASRInput}"` + `rtsp_url:
  "${ASR_RTSP_URL:-rtsp://localhost:8555/live}"` (the local plugin ignores the
  extra key).
- VLM: `rtsp_url: "${VLM_RTSP_URL:-rtsp://localhost:8554/top_camera_raw}"`
  (unchanged default).

`docker-compose.yml` exposes `ASR_INPUT_PLUGIN`, `ASR_RTSP_URL`, `VLM_RTSP_URL`.
To consume the GStreamer stream:

```bash
ASR_INPUT_PLUGIN=GoogleASRRTSPInput \
ASR_RTSP_URL=rtsp://localhost:8555/live \
VLM_RTSP_URL=rtsp://localhost:8555/live \
docker compose up -d om1
```

Both consumers read the same muxed URL; each ffmpeg selects its track
(`google_asr_rtsp` uses `-vn`, `video_rtsp_stream` uses `-an`), so audio and
video come from one synchronized source.

## Code — stamp ASR audio at capture

Previously the audio timestamp was set at *package/send* time
(`packageAudio` → `time.Now().UnixMilli()`), discarding capture timing. Now:

- `packageAudio(pcm, captureMs)` stamps the provided capture time.
- `sendChunkAt(pcm, capture)` added; `sendChunk` kept as a `time.Now()` wrapper
  for back-compat (riva/elevenlabs/parallel unchanged — trivially extendable).
- `google_asr.go` (local mic) and `google_asr_rtsp.go` (RTSP) stamp
  `time.Now()` at the moment the buffer/chunk is read and pass it through.
- Tests updated to pass and assert the capture timestamp.

This makes ASR chunks carry capture time, so downstream alignment with
video-derived features works on a common timeline.

## Deliberately out of scope (follow-ups)

- **Transcript→capture-time mapping across the cloud round-trip.** The final
  transcript still reaches the IO layer without a capture timestamp
  (`asr_common.go` `AddInput(..., time.Time{})`); mapping it back to the source
  audio window needs the ASR WS protocol to echo timestamps.
- **Video RTSP PTS.** JPEG-over-`image2pipe` carries no per-frame PTS; true
  capture-time for video needs a PTS-preserving transport (documented at the
  stamp site in `video_rtsp_stream.go`).
- Extend `sendChunkAt` to the riva/elevenlabs paths (one-line each).

## Verification

Config validated as JSON5; compose validated as YAML. The Go changes are
mechanical (caller consistency checked: `sendChunk` still present; only the two
`packageAudio` test call sites updated) but were **not** compiled in this
environment — run `go build ./... && go test ./plugins/inputs/asr/...` in CI.
