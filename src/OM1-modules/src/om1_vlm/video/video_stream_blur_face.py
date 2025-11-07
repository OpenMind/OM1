import asyncio
import base64
import inspect
import logging
import multiprocessing as mp
import os
import platform
import threading
import time
from queue import Empty, Full
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from om1_utils import LoggingConfig, get_logging_config, setup_logging

from ..anonymizationSys.scrfd_trt_pixelate import (
    CONF_THRES,
    MAX_DETS,
    TOPK_PER_LEVEL,
    TRTInfer,
    apply_pixelation,
    draw_dets,
)
from .video_utils import enumerate_video_devices

try:
    import pycuda.driver as cuda
except ImportError:
    print("pycuda not found, face anonymization will be disabled.")
    cuda = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

SENTINEL = ("__STOP__", None)


def proc_capture(
    out_q: mp.Queue,
    cam: str,
    res: Tuple[int, int],
    fps: int,
    buffer_frames: int,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Capture frames from a camera in a dedicated process and push them to a queue.

    Parameters
    ----------
    out_q : mp.Queue
        Output queue for (timestamp, frame) tuples.
    cam : str
        Camera index/path, e.g. 0 (macOS) or '/dev/video0' (Linux).
    res : Tuple[int, int]
        Requested (width, height).
    fps : int
        Requested FPS.
    buffer_frames : int
        Desired driver-side capture buffer size (best effort).
    logging_config : Optional[LoggingConfig], optional
        logging configuration for this process.
    """
    setup_logging("odom_processor", logging_config=logging_config)
    cv2.setNumThreads(1)
    cap = None

    try:
        # Prefer V4L2 on Linux; macOS ignores this backend id.
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap or not cap.isOpened():
            logging.error(f"[cap] cannot open camera {cam}")
            return

        # Apply settings (best-effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_frames)))
        except Exception:
            pass

        # Warn if camera doesn't honor the requested settings
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (res[0], res[1]):
            logging.warning(
                f"[cap] Camera doesn't support {res}. Using {(actual_w, actual_h)}."
            )

        logging.info(f"[cap] capture loop starting at ~{fps} FPS, device={cam}")
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.error("[cap] error reading frame; retrying shortly.")
                continue

            ts = time.time()
            pkt = (ts, frame)

            # Non-blocking put with drop policy
            try:
                out_q.put_nowait(pkt)
            except Full:
                try:
                    out_q.get_nowait()  # drop one stale
                except Empty:
                    pass
                try:
                    out_q.put_nowait(pkt)
                except Full:
                    pass

    except KeyboardInterrupt:
        logging.info("[cap] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        logging.exception(f"[cap] unexpected error: {e}")
    finally:
        try:
            if cap is not None:
                cap.release()
                logging.info("[cap] released video capture device")
        except Exception:
            pass
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        logging.info("[cap] exit.")


def proc_anonymize(
    in_q: mp.Queue,
    out_q: mp.Queue,
    scrfd_cfg: dict,
    blur_enabled: bool,
    draw_boxes: bool,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Optional anonymizer process: applies face detection & pixelation, then forwards frames.

    Parameters
    ----------
    in_q : mp.Queue
        Input queue receiving (timestamp, frame) tuples from capture process.
    out_q : mp.Queue
        Output queue to send processed (timestamp, frame) tuples to main process.
    scrfd_cfg : dict
        Configuration for the TensorRT SCRFD inferencer and pixelation parameters.
    blur_enabled : bool
        If True and engine_path is provided, apply anonymization.
    draw_boxes : bool
        If True, draw detection boxes for debug.
    logging_config : Optional[LoggingConfig], optional
        logging configuration for this process.
    """
    setup_logging("odom_processor", logging_config=logging_config)
    cv2.setNumThreads(1)

    anonymizer = None
    cuda_ctx = None

    if cuda is None and blur_enabled:
        blur_enabled = False
        logging.error("[anon] pycuda not found, disabling anonymization.")

    try:
        if blur_enabled and scrfd_cfg.get("engine_path"):
            try:
                cuda.init()
                dev_id = int(os.environ.get("OM1_CUDA_DEVICE", "0"))
                cuda_ctx = cuda.Device(dev_id).make_context()
                logging.info(f"[anon] CUDA context created on device {dev_id}")
                anonymizer = _build_anonymizer(scrfd_cfg)
                logging.info("[anon] anonymizer initialized.")
            except Exception:
                try:
                    if cuda_ctx is not None:
                        cuda_ctx.pop()
                except Exception:
                    pass
                logging.exception("[anon] failed to initialize anonymizer.")
                anonymizer = None
                raise

        logging.info(f"[anon] anonymize loop starting (enabled={bool(anonymizer)}).")
        while True:
            try:
                ts, frame = in_q.get(timeout=1.0)
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                logging.info("[anon] received sentinel; exiting loop.")
                break

            if blur_enabled and anonymizer is not None:
                frame, dets, gpu_ms = anonymizer(frame)
                if draw_boxes and dets is not None:
                    draw_dets(frame, dets)

            # pass along processed frame
            try:
                out_q.put_nowait((ts, frame))
            except Full:
                try:
                    out_q.get_nowait()
                except Empty:
                    pass
                try:
                    out_q.put_nowait((ts, frame))
                except Full:
                    pass

    except KeyboardInterrupt:
        logging.info("[anon] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        logging.exception(f"[anon] unexpected error: {e}")
    finally:
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        if cuda_ctx is not None:
            try:
                cuda_ctx.pop()
                logging.info("[anon] CUDA context released.")
            except Exception:
                pass
        logging.info("[anon] exit.")


# ---------------------------------------------------------------------
# Anonymizer Builder
# ---------------------------------------------------------------------
def _build_anonymizer(cfg: dict):
    """
    Build an anonymizer callable that runs SCRFD TensorRT inference
    and applies pixelation to detected faces.

    Parameters
    ----------
    cfg : dict
        Keys include engine_path, input_name, size, conf, topk, max_dets,
        pixel_blocks, pixel_margin, pixel_max_faces, pixel_noise.

    Returns
    -------
    Callable[[any], Tuple[any, Optional[list], Optional[float]]]
        A callable(frame_bgr) -> (frame_bgr, dets, gpu_ms)
    """

    class _Anon:
        def __init__(self, cfg):
            self.inf = TRTInfer(
                engine_path=cfg["engine_path"],
                input_name=cfg.get("input_name", "input.1"),
                size=cfg.get("size", 640),
            )
            self.blocks = int(cfg.get("pixel_blocks", 8))
            self.margin = float(cfg.get("pixel_margin", 0.25))
            self.max_faces = int(cfg.get("pixel_max_faces", 32))
            self.noise = float(cfg.get("pixel_noise", 0.0))
            self.conf = float(cfg.get("conf", CONF_THRES))
            self.topk = int(cfg.get("topk", TOPK_PER_LEVEL))
            self.max_dets = int(cfg.get("max_dets", MAX_DETS))

        def __call__(self, frame_bgr):
            self.inf.conf_thres = self.conf
            self.inf.topk_per_level = self.topk
            self.inf.max_dets = self.max_dets
            dets, gpu_ms = self.inf.infer(frame_bgr)
            if dets is not None and len(dets) > 0:
                apply_pixelation(
                    frame_bgr,
                    dets,
                    margin=self.margin,
                    blocks=self.blocks,
                    max_faces=self.max_faces,
                    noise_sigma=self.noise,
                )
            return frame_bgr, dets, gpu_ms

    return _Anon(cfg)


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
class VideoStreamBlurFace:
    """
    Two-process pipeline:
      - Capture process → pushes (ts, frame_bgr) to q_raw
      - Anonymize process (optional) → pulls from q_raw, pixelates, pushes to q_proc
      - Drain thread (main process) → raw preview callbacks + JPEG/base64 callbacks
    Parameters
    ----------
    frame_callbacks: Callback(s) receiving **base64 JPEG strings** per frame
        (e.g., to send over WebSocket).
    fps: Target output FPS.
    resolution: Frame size `(width, height)` for capture.
    jpeg_quality: JPEG quality (0–100) for encoded outputs.
    device_index: Camera index (e.g., 0, 1) or platform path selector.
    blur_enabled: Enable face anonymization (pixelation) if an engine is provided.
    blur_conf: SCRFD detection confidence threshold.
    scrfd_engine: Path to SCRFD TensorRT engine; if None, disables anonymization.
    scrfd_size: SCRFD model input size (square).
    scrfd_input: SCRFD engine input tensor name.
    pixel_blocks: Pixelation block size (larger = coarser blur).
    pixel_margin: Extra margin around face boxes (fraction of box size).
    pixel_max_faces: Max faces to anonymize per frame.
    pixel_noise: Stddev of Gaussian noise added to pixelated regions (0 = off).
    draw_boxes: Draw detection boxes for debugging.
    queue_size_raw: Max queued frames from capture→anonymize (use 1 to keep freshest).
    queue_size_proc: Max queued frames from anonymize→main (use 1 to keep freshest).
    buffer_frames: Desired capture buffer size (driver hint).
    raw_frame_callbacks: Callback(s) receiving **np.ndarray (BGR)** frames
        per frame (already anonymized but not encoded). Ideal for local preview/recording. Optional.
    """

    def __init__(
        self,
        frame_callbacks: Optional[Iterable[Callable[[str], None]]] = None,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
        jpeg_quality: int = 50,
        device_index: int = 0,
        blur_enabled: bool = True,
        blur_conf: float = 0.5,
        scrfd_engine: Optional[str] = None,
        scrfd_size: int = 640,
        scrfd_input: str = "input.1",
        pixel_blocks: int = 8,
        pixel_margin: float = 0.25,
        pixel_max_faces: int = 32,
        pixel_noise: float = 0.0,
        draw_boxes: bool = False,
        queue_size_raw: int = 1,
        queue_size_proc: int = 1,
        buffer_frames: int = 1,
        raw_frame_callbacks: Optional[Iterable[Callable[[np.ndarray], None]]] = None,
    ):
        self.fps = int(fps)
        cv2.setNumThreads(1)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.encode_quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

        # Callbacks
        self._cb_lock = threading.Lock()
        if frame_callbacks is None:
            self.frame_callbacks: List[Callable[[str], None]] = []
        elif callable(frame_callbacks):
            self.frame_callbacks = [frame_callbacks]  # type: ignore[list-item]
        else:
            self.frame_callbacks = list(frame_callbacks)

        # Raw callbacks (no encode/base64)
        if raw_frame_callbacks is None:
            self.raw_callbacks: List[Callable[[np.ndarray], None]] = []
        elif callable(raw_frame_callbacks):
            self.raw_callbacks = [raw_frame_callbacks]  # type: ignore[list-item]
        else:
            self.raw_callbacks = list(raw_frame_callbacks)

        # Async loop for scheduling async callbacks (if any)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()
        logger.debug("Starting background event loop for video streaming.")

        # Camera device selection
        devices = enumerate_video_devices()
        if platform.system() == "Darwin":
            camindex = 0 if devices else 0
        else:
            camindex = f"/dev/video{devices[0][0]}" if devices else "/dev/video0"

        if device_index != 0:
            if platform.system() == "Darwin":
                camindex = device_index
            else:
                camindex = f"/dev/video{device_index}"

        self.cam = camindex
        self.q_raw = mp.Queue(maxsize=int(queue_size_raw))
        self.q_proc = mp.Queue(maxsize=int(queue_size_proc))

        # Anonymizer config
        self.scrfd_cfg = dict(
            engine_path=scrfd_engine,
            size=int(scrfd_size),
            input_name=scrfd_input,
            conf=float(blur_conf) if blur_conf is not None else CONF_THRES,
            topk=int(TOPK_PER_LEVEL),
            max_dets=int(MAX_DETS),
            pixel_blocks=int(pixel_blocks),
            pixel_margin=float(pixel_margin),
            pixel_max_faces=int(pixel_max_faces),
            pixel_noise=float(pixel_noise),
        )
        self.blur_enabled = bool(blur_enabled)
        self.draw_boxes = bool(draw_boxes)
        self.buffer_frames = int(buffer_frames)

        # Processes and control
        self.p_cap: Optional[mp.Process] = None
        self.p_anon: Optional[mp.Process] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._running = mp.Value("b", False)

    def register_frame_callback(self, cb: Callable[[str], None]) -> None:
        """Register a per-frame (base64 JPEG) callback."""
        if cb is None:
            logger.warning("Frame callback is None, not registering")
            return
        with self._cb_lock:
            if cb not in self.frame_callbacks:
                self.frame_callbacks.append(cb)
                logger.info("Registered new frame callback")
            else:
                logger.warning("Frame callback already registered")

    def register_raw_frame_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """
        Register a per-frame raw (numpy BGR) callback.

        Parameters
        ----------
        cb : Callable[[np.ndarray], None]
            Callback function that takes a single argument: the current video frame in BGR order.
        """
        if cb is None:
            logger.warning("Raw frame callback is None, not registering")
            return
        with self._cb_lock:
            if cb not in self.raw_callbacks:
                self.raw_callbacks.append(cb)
                logger.info("Registered new raw frame callback")
            else:
                logger.warning("Raw frame callback already registered")

    def unregister_frame_callback(self, cb: Callable[[str], None]) -> None:
        """
        Unregister a previously added base64 JPEG callback.

        Parameters
        ----------
        cb : Callable[[str], None]
            The callback function to unregister.
        """
        with self._cb_lock:
            try:
                self.frame_callbacks.remove(cb)
                logger.info("Unregistered frame callback")
            except ValueError:
                logger.warning("Attempted to unregister a non-registered callback")

    def unregister_raw_frame_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """
        Unregister a previously added raw frame callback.

        Parameters
        ----------
        cb : Callable[[np.ndarray], None]
            The raw frame callback function to unregister.
        """
        with self._cb_lock:
            try:
                self.raw_callbacks.remove(cb)
                logger.info("Unregistered raw frame callback")
            except ValueError:
                logger.warning("Attempted to unregister a non-registered callback")

    def start(self) -> None:
        """
        Start capture/anonymize worker processes and the drain thread.
        """
        if self._running.value:
            logger.warning("[main] start() called but already running.")
            return

        self._running.value = True
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

        self.p_cap = mp.Process(
            target=proc_capture,
            args=(
                self.q_raw,
                self.cam,
                self.resolution,
                self.fps,
                self.buffer_frames,
                get_logging_config(),
            ),
            daemon=True,
            name="CaptureProc",
        )
        self.p_anon = mp.Process(
            target=proc_anonymize,
            args=(
                self.q_raw,
                self.q_proc,
                self.scrfd_cfg,
                self.blur_enabled,
                self.draw_boxes,
                get_logging_config(),
            ),
            daemon=True,
            name="AnonProc",
        )

        self.p_cap.start()
        logger.info(f"[main] Capture process started (pid={self.p_cap.pid}).")
        self.p_anon.start()
        logger.info(f"[main] Anonymize process started (pid={self.p_anon.pid}).")

        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="DrainThread"
        )
        self._drain_thread.start()
        logger.info("[main] Drain thread started. VideoStream running.")

    def stop(self, join_timeout: float = 1.0) -> None:
        """
        Stop workers and drain thread, and release resources.

        Parameters
        ----------
        join_timeout : float, optional
            Time (seconds) to wait for processes/threads to join before terminate, by default 1.0
        """
        if not self._running.value:
            logger.warning("[main] stop() called but not running.")
            return

        self._running.value = False

        # Signal workers
        for q in (self.q_raw, self.q_proc):
            try:
                q.put_nowait(SENTINEL)
            except Full:
                pass

        # Join processes
        if self.p_cap and self.p_cap.is_alive():
            self.p_cap.join(timeout=join_timeout)
            if self.p_cap.is_alive():
                logger.warning("[main] CaptureProc did not exit in time; terminating.")
                self.p_cap.terminate()

        if self.p_anon and self.p_anon.is_alive():
            self.p_anon.join(timeout=join_timeout)
            if self.p_anon.is_alive():
                logger.warning("[main] AnonProc did not exit in time; terminating.")
                self.p_anon.terminate()

        # Join drain thread
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=join_timeout)

        # Stop background loop
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

        logger.info("[main] VideoStream stopped.")

    # ----------------------------
    # Internals
    # ----------------------------
    def _start_loop(self) -> None:
        """
        Start the asyncio event loop in a dedicated thread.
        """
        asyncio.set_event_loop(self.loop)
        logger.debug("Starting background asyncio event loop.")
        self.loop.run_forever()

    def _dispatch(self, b64_jpeg: str) -> None:
        """
        Dispatch a base64-encoded JPEG frame to all registered callbacks.

        Parameters
        ----------
        b64_jpeg : str
            Base64-encoded JPEG bytes.
        """
        with self._cb_lock:
            callbacks = tuple(self.frame_callbacks)
        for cb in callbacks:
            try:
                result = cb(b64_jpeg)
                if inspect.isawaitable(result):
                    asyncio.run_coroutine_threadsafe(result, self.loop)
            except Exception as e:
                logger.error(f"[main] Frame callback raised: {e}", exc_info=True)

    def _dispatch_raw(self, frame_bgr: np.ndarray) -> None:
        """
        Dispatch a raw numpy frame to raw callbacks (no encode/base64).

        Parameters
        ----------
        frame_bgr : np.ndarray
            The current video frame in BGR order.
        """
        with self._cb_lock:
            raw_cbs = tuple(self.raw_callbacks)
        for cb in raw_cbs:
            try:
                cb(frame_bgr)
            except Exception as e:
                logger.error(f"[main] Raw frame callback raised: {e}", exc_info=True)

    def _drain_loop(self) -> None:
        """
        Main-process drain loop:
        - Pull processed frames, drain queue to newest
        - Dispatch raw frame (no encoding) for local preview (optional)
        - JPEG encode + base64 for standard callbacks
        - Logs output FPS and latency stats
        """

        while self._running.value:
            try:
                ts, frame = self.q_proc.get(timeout=max(0.005, 1.0 / max(1, self.fps)))
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                logger.info("[main] drain loop received sentinel; exiting.")
                break

            self._dispatch_raw(frame)

            try:
                _, buffer = cv2.imencode(".jpg", frame, self.encode_quality)
                self._dispatch(base64.b64encode(buffer).decode("utf-8"))
            except Exception as e:
                logger.error(
                    f"[main] error encoding/dispatching frame: {e}", exc_info=True
                )
                continue
