# Video Test System

Goal: Try out alternative system for sharing video and audio data with LLMs and human teleoperators.

Key idea: use a central media server (like MediaMTX) to handle the streams and codecs.

[MediaMTX](https://github.com/bluenviron/mediamtx) is a ready-to-use and zero-dependency real-time media server and media proxy that allows to publish, read, proxy, record and playback video and audio streams.

ToDo: use the `ffmpeg` in the Docker image?

## Basic Setup on Mac

### Start the MediaMTX server

Start `Docker.app` on your Mac. Then, run the `bluenviron/mediamtx:latest-ffmpeg`:

```bash
docker run --rm -it -p 8554:8554 -p 1935:1935 -p 8889:8889 -p 8189:8189/udp bluenviron/mediamtx:latest-ffmpeg
```

The point of the `-p 8554:8554` is to make the docker container RTSP listener port (on :8554 (TCP), :8000 (UDP/RTP), :8001 (UDP/RTCP)) available to the mac. `-p 1935:1935` allows access to the RTMP listener.

### Stream video data to the MediaMTX server

Note that the MediaMTX could live anywhere, for example in the cloud, but for testing let's have it run locally.

List all devices on a Mac:

```bash
ffmpeg -hide_banner -list_devices true -f avfoundation -i dummy
```
(will also give error message but that's ok)

```
[AVFoundation indev @ 0x13c606250] AVFoundation video devices:
[AVFoundation indev @ 0x13c606250] [0] Studio Display Camera
[AVFoundation indev @ 0x13c606250] [1] FaceTime HD Camera
[AVFoundation indev @ 0x13c606250] [2] Capture screen 0
[AVFoundation indev @ 0x13c606250] AVFoundation audio devices:
[AVFoundation indev @ 0x13c606250] [0] MacBook Air Microphone
[AVFoundation indev @ 0x13c606250] [1] Studio Display Microphone
```

Then, start sending video data to the local `mediamtx` at `rtmp://localhost:1935/live`:

```bash
ffmpeg -f avfoundation -video_size 1920x1080 -framerate 30 -i "0:0" -vcodec libx264 -preset ultrafast -tune zerolatency -f flv "rtmp://localhost:1935/live"
```

-i "0:0": Specifies the input device. In this case, 0 refers to the video device index and the second 0 refers to the audio device index from the listed devices. Adjust these indices based on the output of the -list_devices command.

`ffmpeg` should report a working stream:

```bash
Output #0, flv, to 'rtmp://localhost:1935/live':
  Metadata:
    encoder         : Lavf61.7.100
  Stream #0:0: Video: h264 ([7][0][0][0] / 0x0007), yuv422p(tv, progressive), 1920x1080, q=2-31, 1000k fps, 1k tbn
      Metadata:
        encoder         : Lavc61.19.101 libx264
      Side data:
        cpb: bitrate max/min/avg: 0/0/0 buffer size: 0 vbv_delay: N/A
  Stream #0:1: Audio: mp3 ([2][0][0][0] / 0x0002), 48000 Hz, mono, fltp
      Metadata:
        encoder         : Lavc61.19.101 libmp3lame
frame= 3711 fps= 30 q=26.0 size=   38238KiB time=00:02:03.84 bitrate=2529.4kbits/s speed=0.999x
```

The MediaMTX should report:

```bash
2025/09/16 23:28:06 INF [RTMP] [conn 192.168.65.1:64081] opened
2025/09/16 23:28:08 INF [RTMP] [conn 192.168.65.1:64081] is publishing to path 'live', 2 tracks (H264, MPEG-1/2 Audio)
```

To stream audio using the **opus codec**, upgrade your **FFmpeg** to version **8.x.x**. Once updated, use the following command to stream it—this will enable WebRTC support.

```bash
ffmpeg -f avfoundation -video_size 640x480 -framerate 30 -i "0:0" -c:v libx264 -pix_fmt yuv420p -preset ultrafast -b:v 600k -c:a libopus -ar 48000 -ac 2 -b:a 128k -f flv "rtmp://localhost:1935/live"
```

### Consume the data from the MediaMTX server

You can access the data using dozens of protocols or apps. For example, to use VLC and `rtsp`, install VLC, go to `File -> Open Network` and enter `rtsp://localhost:8554/live`.

**Note** - it may take a few seconds for the stream to open. A typical delay is about 2 seconds.

You can also use WebRTC to view the video in your web browser by visiting http://localhost:8889/live.

### OM Remote Server

You can stream your video to our remote server using your **OM API Key**:

```bash
ffmpeg -f avfoundation -video_size 1920x1080 -framerate 30 -i "0:0" \
  -vcodec libx264 -preset ultrafast -tune zerolatency -f flv \
  "rtmp://api-video-ingest.openmind.org:1935/<OM_API_KEY_ID>?api_key=<OM_API_KEY>"
```

**Note:** **OM_API_KEY_ID** refers to the first 16 digits of your API key, excluding the **om_prod_ prefix**. You can also find your corresponding **OM_API_KEY_ID** in our [portal](https://portal.openmind.org).

You can view your video stream at:

```bash
https://api-video-webrtc.openmind.org/<OM_API_KEY_ID>?api_key=<OM_API_KEY>
```
