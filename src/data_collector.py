import logging
import multiprocessing as mp
import os
import signal
import struct
import sys
import threading
import time
import wave
from datetime import datetime, timezone

import cv2
import numpy as np


def _record_video(rtsp_url: str, stop_event: threading.Event, rollover_seconds: int):
    logging.info(f"DataCollector: Starting video recording from {rtsp_url}")
    
    # Reduce OpenCV connection timeout (ms) to prevent hanging for 30s on disconnect, 
    # and force TCP protocol (to bypass macOS/Docker UDP packet loss issues)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000"

    cap = None
    while not stop_event.is_set():
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            break
        logging.warning(f"DataCollector: Could not open video RTSP. Retrying in 5s...")
        cap.release()
        time.sleep(5)

    if stop_event.is_set() or cap is None or not cap.isOpened():
        return

    # Read one frame to get EXACT dimensions natively (RTSP .get() sometimes returns 0)
    ret, frame = cap.read()
    if not ret or frame is None:
        logging.error("DataCollector: Cap opened but failed to read first frame.")
        cap.release()
        return

    height, width = frame.shape[:2]
    fps = 30.0
    logging.info(f"DataCollector: Video RTSP stream opened. Dim: {width}x{height}")

    os.makedirs("recordings", exist_ok=True)
    
    file_start_time = time.time()
    def _open_video_writer():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        mp4_path = f"recordings/data_collector_video_{timestamp}.mp4"
        avi_path  = f"recordings/data_collector_video_{timestamp}.avi"

        # ── 1. GStreamer + Jetson NVENC (nvv4l2h264enc) ────────────────────────
        # appsrc feeds raw BGR frames; nvvidconv converts colour space on the
        # GPU; nvv4l2h264enc is the Jetson hardware H.264 encoder (no CPU load);
        # mp4mux + filesink write the output file.
        # cv2.VideoWriter with CAP_GSTREAMER accepts the pipeline as the filename
        # and fourcc=0.
        gst_pipeline = (
            f"appsrc ! "
            f"video/x-raw,format=BGR,width={width},height={height},"
            f"framerate={int(fps)}/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGRx ! "
            f"nvvidconv ! "
            f"video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate=8000000 iframeinterval=30 ! "
            f"h264parse ! "
            f"mp4mux ! "
            f"filesink location={mp4_path}"
        )
        w = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        if w.isOpened():
            logging.info(f"DataCollector: VideoWriter opened via GStreamer nvv4l2h264enc ({mp4_path})")
            return w
        w.release()
        logging.warning("DataCollector: GStreamer nvv4l2h264enc unavailable, falling back to OpenCV codecs...")

        # ── 2–4. OpenCV codec fallback chain ──────────────────────────────────
        #   avc1 – H.264 via V4L2M2M (generic ARM hardware encoder)
        #   mp4v – software MPEG-4  (most FFmpeg builds)
        #   MJPG – Motion JPEG .avi (always available, no external deps)
        codecs = [
            (cv2.VideoWriter_fourcc(*"avc1"), mp4_path),
            (cv2.VideoWriter_fourcc(*"mp4v"), mp4_path),
            (cv2.VideoWriter_fourcc(*"MJPG"), avi_path),
        ]
        for fourcc, filepath in codecs:
            w = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            if w.isOpened():
                logging.info(f"DataCollector: VideoWriter opened ({filepath})")
                return w
            w.release()
            logging.warning(f"DataCollector: Codec {fourcc:#010x} rejected, trying next fallback...")

        logging.error("DataCollector: All video codecs failed. Video recording disabled.")
        return None



    writer = _open_video_writer()

    if writer is None or not writer.isOpened():
        cap.release()
        return

    # Write the first grabbed frame
    writer.write(frame)

    try:
        while not stop_event.is_set():
            # Rollover file every `rollover_seconds`
            if time.time() - file_start_time >= rollover_seconds:
                writer.release()
                # Give the kernel a moment to fully reclaim the V4L2M2M encoder
                # device before attempting to re-open it. Without this, the
                # immediate re-open races the driver teardown and fails.
                time.sleep(0.5)
                writer = _open_video_writer()
                if writer is None or not writer.isOpened():
                    break
                file_start_time = time.time()
                
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
            else:
                logging.warning("DataCollector: Video stream lost, waiting for frame...")
                time.sleep(1.0)
    finally:
        writer.release()
        cap.release()
        logging.info("DataCollector: Video recording stopped and file saved gracefully.")


def _record_audio(rtsp_url: str, stop_event: threading.Event, rollover_seconds: int):
    logging.info(f"DataCollector: Starting audio recording from {rtsp_url}")
    try:
        from om1_speech import AudioRTSPInputStream
    except ImportError:
        logging.error("DataCollector: om1_speech not found for audio recording")
        return

    rate = 16000
    os.makedirs("recordings", exist_ok=True)
    
    class AudioState:
        def __init__(self):
            self.lock = threading.Lock()
            self.file_start_time = time.time()
            self.wav_file = self._open_wav()

        def _open_wav(self):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            wf = wave.open(f"recordings/data_collector_audio_{timestamp}.wav", "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            return wf

        def check_rollover(self):
            with self.lock:
                if time.time() - self.file_start_time >= rollover_seconds:
                    self.wav_file.close()
                    self.wav_file = self._open_wav()
                    self.file_start_time = time.time()

        def write(self, data):
            with self.lock:
                self.wav_file.writeframes(data)

        def close(self):
            with self.lock:
                self.wav_file.close()

    state = AudioState()

    def audio_callback(data):
        state.check_rollover()
        try:
            # Handle possible string wrapped json or raw bytes depending on om1_speech config
            if isinstance(data, bytes):
                state.write(data)
            elif isinstance(data, str):
                import base64
                import json

                payload = json.loads(data)
                if "audio" in payload:
                    state.write(base64.b64decode(payload["audio"]))
        except Exception:
            pass

    audio_stream = AudioRTSPInputStream(
        rtsp_url=rtsp_url, rate=rate, chunk=1024, audio_data_callback=audio_callback
    )
    
    try:
        audio_stream.start()
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        audio_stream.stop()
        state.close()
        logging.info("DataCollector: Audio recording stopped and saved gracefully.")


def _record_lidar_zenoh(topic: str, stop_event: threading.Event, rollover_seconds: int):
    logging.info(f"DataCollector: Starting 2D LiDAR (Zenoh) recording on topic: {topic}")
    try:
        import DracoPy
    except ImportError:
        logging.error("DataCollector: DracoPy is not installed. Zenoh Lidar recording disabled.")
        return

    try:
        import zenoh
        from zenoh_msgs import LaserScan, open_zenoh_session, sensor_msgs
    except ImportError:
        logging.error("DataCollector: zenoh or zenoh_msgs not found. Zenoh Lidar recording disabled.")
        return

    os.makedirs("recordings", exist_ok=True)
    def _open_lidar():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return open(f"recordings/data_collector_2d_zenoh_lidar_{timestamp}.drcs", "wb")

    lidar_file = _open_lidar()
    file_start_time = time.time()
    last_data_log_time = time.time()
    has_warned_empty = False

    class ZenohHandler:
        def __init__(self):
            self.lock = threading.Lock()
            self.msg = None
            
        def listen_scan(self, data: zenoh.Sample):
            try:
                msg = sensor_msgs.LaserScan.deserialize(data.payload.to_bytes())
                with self.lock:
                    self.msg = msg
            except Exception as e:
                logging.debug(f"Zenoh LaserScan deserialize error: {e}")

    handler = ZenohHandler()
    try:
        zen = open_zenoh_session()
        logging.info(f"DataCollector: Zenoh session opened. Declaring subscriber on {topic}")
        sub = zen.declare_subscriber(topic, handler.listen_scan)
    except Exception as e:
        logging.error(f"DataCollector: Error opening Zenoh subscriber: {e}")
        lidar_file.close()
        return

    try:
        while not stop_event.is_set():
            if time.time() - file_start_time >= rollover_seconds:
                lidar_file.flush()
                os.fsync(lidar_file.fileno())
                lidar_file.close()
                lidar_file = _open_lidar()
                file_start_time = time.time()
                
            msg_to_process = None
            with handler.lock:
                if handler.msg is not None:
                    msg_to_process = handler.msg
                    handler.msg = None
                    
            if msg_to_process:
                has_warned_empty = False
                last_data_log_time = time.time()
                
                try:
                    angles = np.arange(
                        msg_to_process.angle_min,
                        msg_to_process.angle_max,
                        msg_to_process.angle_increment,
                        dtype=np.float32,
                    )
                    ranges = np.array(msg_to_process.ranges, dtype=np.float32)
                    n = min(len(angles), len(ranges))
                    angles = angles[:n]
                    ranges = ranges[:n]

                    # Filter invalid / out-of-range readings
                    valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < 12.0)
                    angles = angles[valid]
                    ranges = ranges[valid]

                    if len(ranges) > 0:
                        # Convert polar → Cartesian XYZ (Z=0 for 2-D scan)
                        pts = np.zeros((len(ranges), 3), dtype=np.float32)
                        pts[:, 0] = ranges * np.cos(angles)
                        pts[:, 1] = ranges * np.sin(angles)

                        compressed = DracoPy.encode(pts)
                        lidar_file.write(struct.pack("<I", len(compressed)))
                        lidar_file.write(compressed)
                        lidar_file.flush()
                        os.fsync(lidar_file.fileno())
                except Exception as e:
                    logging.error(f"DataCollector Zenoh 2D Lidar encoding error: {e}")
            
            if time.time() - last_data_log_time > 10.0:
                if not has_warned_empty:
                    logging.warning(f"DataCollector: No Zenoh LaserScan received on '{topic}' for over 10s.")
                    has_warned_empty = True
                last_data_log_time = time.time()
                
            time.sleep(0.05)
    except Exception as e:
        logging.error(f"DataCollector Zenoh 2D LiDAR loop exception: {e}")
    finally:
        lidar_file.flush()
        lidar_file.close()
        if 'sub' in locals():
            try:
                sub.undeclare()
            except Exception:
                pass
        logging.info("DataCollector: Zenoh 2D LiDAR recording stopped gracefully.")


def _record_odom(topic: str, stop_event: threading.Event, rollover_seconds: int):
    import json
    import math

    logging.info(f"DataCollector: Starting Odom (Zenoh) recording on topic: {topic}")
    try:
        import zenoh
        from zenoh_msgs import Odometry, open_zenoh_session, nav_msgs
    except ImportError:
        logging.error("DataCollector: zenoh or zenoh_msgs not found. Odom recording disabled.")
        return

    os.makedirs("recordings", exist_ok=True)
    def _open_odom():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return open(f"recordings/data_collector_odom_{timestamp}.jsonl", "w")

    odom_file = _open_odom()
    file_start_time = time.time()
    last_data_log_time = time.time()
    has_warned_empty = False

    class OdomHandler:
        def __init__(self):
            self.lock = threading.Lock()
            self.msg = None

        def listen_odom(self, data: zenoh.Sample):
            try:
                msg = nav_msgs.Odometry.deserialize(data.payload.to_bytes())
                with self.lock:
                    self.msg = msg
            except Exception as e:
                logging.debug(f"Zenoh Odometry deserialize error: {e}")

    handler = OdomHandler()
    try:
        zen = open_zenoh_session()
        logging.info(f"DataCollector: Zenoh session opened. Declaring odom subscriber on {topic}")
        sub = zen.declare_subscriber(topic, handler.listen_odom)
    except Exception as e:
        logging.error(f"DataCollector: Error opening Zenoh odom subscriber: {e}")
        odom_file.close()
        return

    try:
        while not stop_event.is_set():
            if time.time() - file_start_time >= rollover_seconds:
                odom_file.flush()
                os.fsync(odom_file.fileno())
                odom_file.close()
                odom_file = _open_odom()
                file_start_time = time.time()

            msg_to_process = None
            with handler.lock:
                if handler.msg is not None:
                    msg_to_process = handler.msg
                    handler.msg = None

            if msg_to_process:
                has_warned_empty = False
                last_data_log_time = time.time()

                try:
                    p = msg_to_process.pose.pose.position
                    q = msg_to_process.pose.pose.orientation
                    lv = msg_to_process.twist.twist.linear
                    av = msg_to_process.twist.twist.angular
                    hdr = msg_to_process.header

                    # Quaternion → yaw (rotation around Z axis)
                    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    yaw_rad = math.atan2(siny_cosp, cosy_cosp)

                    record = {
                        "ts": time.time(),
                        "ros_ts": hdr.stamp.sec + hdr.stamp.nanosec * 1e-9,
                        "pos_x": p.x,
                        "pos_y": p.y,
                        "pos_z": p.z,
                        "quat_x": q.x,
                        "quat_y": q.y,
                        "quat_z": q.z,
                        "quat_w": q.w,
                        "yaw_rad": yaw_rad,
                        "yaw_deg": math.degrees(yaw_rad),
                        "lin_vx": lv.x,
                        "lin_vy": lv.y,
                        "lin_vz": lv.z,
                        "ang_vz": av.z,
                    }
                    odom_file.write(json.dumps(record) + "\n")
                    odom_file.flush()
                    os.fsync(odom_file.fileno())
                except Exception as e:
                    logging.error(f"DataCollector Odom encoding error: {e}")

            if time.time() - last_data_log_time > 10.0:
                if not has_warned_empty:
                    logging.warning(f"DataCollector: No Zenoh Odometry received on '{topic}' for over 10s.")
                    has_warned_empty = True
                last_data_log_time = time.time()

            time.sleep(0.02)  # ~50 Hz polling
    except Exception as e:
        logging.error(f"DataCollector Zenoh Odom loop exception: {e}")
    finally:
        odom_file.flush()
        odom_file.close()
        if 'sub' in locals():
            try:
                sub.undeclare()
            except Exception:
                pass
        logging.info("DataCollector: Odom recording stopped gracefully.")


def run_data_collector_process(video_rtsp: str, audio_rtsp: str, lidar_topic: str, odom_topic: str = "**/odom", rollover_seconds: int = 120):
    """
    Main entry point for the data collector process.
    Spawned via multiprocessing.Process in run.py.
    """
    stop_event = threading.Event()

    def handle_exit(signum, frame):
        logging.info(f"DataCollector: Received termination signal ({signum}), shutting down safely...")
        stop_event.set()

    # Catch Ctrl+C (SIGINT) and termination (SIGTERM)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    threads = []

    if video_rtsp:
        t = threading.Thread(target=_record_video, args=(video_rtsp, stop_event, rollover_seconds), daemon=True)
        t.start()
        threads.append(t)

    if audio_rtsp:
        t = threading.Thread(target=_record_audio, args=(audio_rtsp, stop_event, rollover_seconds), daemon=True)
        t.start()
        threads.append(t)

    if lidar_topic:
        t_zenoh = threading.Thread(target=_record_lidar_zenoh, args=(lidar_topic, stop_event, rollover_seconds), daemon=True)
        t_zenoh.start()
        threads.append(t_zenoh)
    
    if odom_topic:
        t = threading.Thread(target=_record_odom, args=(odom_topic, stop_event, rollover_seconds), daemon=True)
        t.start()
        threads.append(t)

    try:
        # Loop until stop_event is set via signals
        while not stop_event.is_set():
            time.sleep(1)
    except Exception:
        pass
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)
        logging.info("DataCollector: All child threads joined, process exiting cleanly.")
        sys.exit(0)
