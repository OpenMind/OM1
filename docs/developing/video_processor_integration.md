# Video Processor Integration (audio & video sources)

On a robot (e.g. NVIDIA Thor), OM1 runs alongside the **OM1 Video Processor**, a
separate container that owns the camera and microphone. This page explains how
OM1 consumes that data, and — importantly — the trade-offs behind the defaults
so you can change them deliberately later.

## Topology

The video-processor is the capture authority. Its GStreamer pipeline captures
camera + mic on one clock and serves, on the robot host:

| Endpoint | Contents | Who it's for |
| --- | --- | --- |
| `rtsp://localhost:8555/live` | Processed video (recognition boxes, blur) **+ audio**, synchronized | machine consumers (OM1, cloud relay) — low latency |
| `rtsp://localhost:8556/raw` | **Raw** camera view (no overlays/blur), video only | machine consumers wanting a clean image |
| `mediamtx` `:8554` (RTSP) / `:8888` (HLS) / `:8889` (WebRTC) | Re-serves `/live` and `/raw` | people / external clients / browsers |

**OM1 consumes the gst endpoints directly (`:8555`/`:8556`), not mediamtx.**
mediamtx is an on-demand fan-out hub for humans; routing the agent's real-time
inputs through it would add a relay hop and couple OM1's core perception to an
optional convenience service. Keep the agent on the direct, low-latency source;
leave mediamtx for people.

## How OM1 consumes it

Set via env (see `docker-compose.yml` and `config/unitree_g1_conversation.json5`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ASR_INPUT_PLUGIN` | `GoogleASRInput` | ASR source plugin (local mic vs RTSP) |
| `ASR_RTSP_URL` | `rtsp://localhost:8555/live` | Audio source when using the RTSP ASR plugin (ffmpeg selects the audio track with `-vn`) |
| `VLM_RTSP_URL` | `rtsp://localhost:8556/raw` | Video source for the VLM |

### VLM video source — why `/raw`

The VLM describes the scene for the LLM. `/live` has recognition boxes and
**blurred faces** burned into the pixels; feeding that to the VLM degrades its
descriptions. `/raw` is the clean camera view, so it gives the best scene
understanding — hence the default.

Trade-off to know: `/raw` is **not** anonymized. If your deployment must keep
faces blurred even in what's sent to the (cloud) VLM for privacy reasons, set
`VLM_RTSP_URL=rtsp://localhost:8555/live` instead and accept the description
quality hit.

### ASR source — why local mic is the default

ASR is env-selectable between the local mic (`GoogleASRInput`, opens the mic via
PortAudio) and RTSP (`GoogleASRRTSPInput`, pulls audio from `:8555/live`). The
default is the **local mic**. Reasoning:

- **Standalone-safe.** OM1 runs without the video-processor (dev laptops, other
  robots, `conversation.json5`). A local-mic default doesn't break when there's
  no `:8555`.
- **Lower latency & decoupled.** Direct PortAudio capture avoids the RTSP +
  ffmpeg decode hop, and OM1's hearing doesn't depend on the video-processor
  being up.
- **The double mic capture is harmless** (see below).

Switch to RTSP (`ASR_INPUT_PLUGIN=GoogleASRRTSPInput`, `ASR_RTSP_URL=
rtsp://localhost:8555/live`) when you specifically want a **single audio
authority** — one capture, one AEC path, and OM1's transcripts aligned to the
*exact* audio that the video-processor records and streams to the cloud. The
cost is coupling OM1's ASR to the video-processor (startup ordering; goes deaf
if `:8555` is down) and a small latency hop.

Rule of thumb: timing-critical / sample-consistent audio work belongs in the
video-processor (which already has the single capture-stamped audio for
recording, features, and cloud). OM1's own job — hear the user, respond fast —
is best served by the decoupled local mic. Flip to RTSP only when a concrete
need for shared/consistent audio appears.

## Can OM1 and the video-processor both use the microphone at once?

**Yes — because access goes through PulseAudio.** `default_mic_aec` is a
PulseAudio (echo-cancelled) *virtual source*, and PulseAudio sources are not
exclusive: it duplicates the stream to every client. Both containers mount the
host Pulse socket (`PULSE_SERVER=unix:$XDG_RUNTIME_DIR/pulse/native`), so OM1's
PortAudio→Pulse capture and the video-processor's `pulsesrc`/ffmpeg capture read
the same source concurrently without conflict.

Caveat: this only holds via PulseAudio. If either side opened the **raw ALSA**
device (`hw:X`) directly, ALSA hardware devices are exclusive and the second
opener would fail with "device busy" (unless using `dmix`/`dsnoop`). Everything
here is Pulse-based, so concurrent access is fine.

## Video timestamps in OM1

OM1's RTSP video consumer stamps each frame with local **receive time**
(`time.Now()`), not the original capture time — the JPEG-over-pipe transport
carries no per-frame PTS. This is intentional: OM1 fuses VLM output as coarse
"recent context", so sub-second pipeline jitter is irrelevant, and any
frame-accurate A/V work lives in the video-processor (capture-stamped at the
source). If OM1 ever needs true capture time, switch the consumer to an RTSP
client that exposes RTP/RTCP timing (e.g. `gortsplib`) instead of the
ffmpeg→JPEG pipe. See `internal/providers/vlm/video_rtsp_stream.go`.
