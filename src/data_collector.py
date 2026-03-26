import logging
import multiprocessing as mp
import os
import signal
import struct
import sys
import threading
import time
import wave
from datetime import datetime

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"recordings/data_collector_video_{timestamp}.mp4"
        
        # Try H264 first (avc1)
        w = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not w.isOpened():
            logging.warning("DataCollector: avc1 codec rejected by cv2 Linux backend. Falling back to mp4v...")
            w = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            
        if not w.isOpened():
            logging.error(f"DataCollector: FATAL ERROR. Could not open VideoWriter for {filepath}")
        else:
            logging.info(f"DataCollector: VideoWriter successfully initialized for {filepath}")
        return w

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
                writer = _open_video_writer()
                if not writer.isOpened():
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _record_lidar(serial_port: str, stop_event: threading.Event, rollover_seconds: int):
    logging.info(f"DataCollector: Starting LiDAR recording on {serial_port}")
    try:
        import DracoPy
    except ImportError:
        logging.error("DataCollector: DracoPy is not installed. Lidar disabled.")
        return

    try:
        from providers.rplidar_driver import RPDriver
    except ImportError:
        logging.error("DataCollector: RPDriver not found")
        return

    os.makedirs("recordings", exist_ok=True)
    def _open_lidar():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return open(f"recordings/data_collector_lidar_{timestamp}.drcs", "wb")

    lidar_file = _open_lidar()
    file_start_time = time.time()
    last_data_log_time = time.time()
    has_warned_empty = False

    lidar = None
    for baud in [256000, 115200, 1000000]:
        try:
            logging.info(f"DataCollector: Attempting LiDAR connection on {serial_port} @ {baud}")
            lidar = RPDriver(serial_port, baudrate=baud)
            lidar.get_info()
            break
        except Exception:
            if lidar:
                try:
                    lidar.disconnect()
                except:
                    pass
            lidar = None

    if not lidar:
        logging.error(f"DataCollector: LiDAR failed to connect to {serial_port} at all tested baud rates.")
        lidar_file.close()
        return

    try:
        lidar.reset()
        time.sleep(0.5)

        for scan in lidar.iter_scans_local(scan_type="express", max_buf_meas=0, min_len=5, max_distance_mm=10000):
            if stop_event.is_set():
                break

            if time.time() - file_start_time >= rollover_seconds:
                lidar_file.flush()
                os.fsync(lidar_file.fileno())
                lidar_file.close()
                lidar_file = _open_lidar()
                file_start_time = time.time()
                
            has_warned_empty = False
            last_data_log_time = time.time()

            if len(scan) > 0:
                try:
                    scan_array = np.array(scan)
                    # RPDriver returns (angle in degrees, distance in mm)
                    angles_deg = scan_array[:, 0]
                    distances_m = scan_array[:, 1] / 1000.0
                    
                    # Convert to cartesian coordinates
                    angles_rad = angles_deg * (math.pi / 180.0)
                    pts = np.zeros((len(scan), 3), dtype=np.float32)
                    pts[:, 0] = distances_m * np.cos(angles_rad)
                    pts[:, 1] = distances_m * np.sin(angles_rad)
                    
                    compressed = DracoPy.encode(pts)
                    lidar_file.write(struct.pack("<I", len(compressed)))
                    lidar_file.write(compressed)
                    lidar_file.flush()
                    os.fsync(lidar_file.fileno())  # Force write to physical disk
                except Exception as e:
                    logging.error(f"DataCollector Lidar encoding error: {e}")

            # Also check heartbeat inside loop defensively
            if time.time() - last_data_log_time > 10.0:
                if not has_warned_empty:
                    logging.warning(f"DataCollector: No LiDAR data received on '{serial_port}' for over 10s.")
                    has_warned_empty = True
                last_data_log_time = time.time()

    except Exception as e:
        logging.error(f"DataCollector LiDAR loop exception: {e}")
    finally:
        lidar_file.flush()
        lidar_file.close()
        if lidar:
            try:
                lidar.stop()
                lidar.disconnect()
            except Exception:
                pass
        logging.info("DataCollector: LiDAR recording stopped and saved gracefully.")


def _record_odom(channel: str, stop_event: threading.Event, rollover_seconds: int):
    logging.info(f"DataCollector: Starting Odom (Yaw) recording on {channel}")
    try:
        from providers.unitree_go2_odom_provider import UnitreeGo2OdomProvider
    except ImportError:
        logging.error("DataCollector: Odom provider not found")
        return

    odom = UnitreeGo2OdomProvider(channel=channel)

    os.makedirs("recordings", exist_ok=True)
    def _open_odom():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return open(f"recordings/data_collector_odom_{timestamp}.jsonl", "w")

    odom_file = _open_odom()
    file_start_time = time.time()

    try:
        while not stop_event.is_set():
            if time.time() - file_start_time >= rollover_seconds:
                odom_file.flush()
                os.fsync(odom_file.fileno())
                odom_file.close()
                odom_file = _open_odom()
                file_start_time = time.time()
                
            if hasattr(odom, "odom_yaw_m180_p180"):
                import json
                data = {
                    "ts": time.time(),
                    "yaw_m180_p180": odom.odom_yaw_m180_p180,
                    "yaw_0_360": odom.odom_yaw_0_360,
                    "x": getattr(odom, "x", 0.0),
                    "y": getattr(odom, "y", 0.0)
                }
                odom_file.write(json.dumps(data) + "\n")
                odom_file.flush()
                os.fsync(odom_file.fileno())  # Force write to physical disk to prevent data loss on sudden power failure
            time.sleep(0.05)  # 20Hz polling for yaw
    finally:
        odom_file.flush()
        odom_file.close()
        logging.info("DataCollector: Odom (Yaw) recording stopped and saved gracefully.")


def run_data_collector_process(video_rtsp: str, audio_rtsp: str, lidar_port: str, odom_channel: str = "eth0", rollover_seconds: int = 120):
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

    if lidar_port:
        t = threading.Thread(target=_record_lidar, args=(lidar_port, stop_event, rollover_seconds), daemon=True)
        t.start()
        threads.append(t)

    if odom_channel:
        t = threading.Thread(target=_record_odom, args=(odom_channel, stop_event, rollover_seconds), daemon=True)
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
