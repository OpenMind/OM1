# -*- coding: utf-8 -*-
"""
SCRFD TensorRT on Jetson Orin + Pixelation anonymization
- TRT 10.x tensor API (set_tensor_address + execute_async_v3)
- Pinned host buffers, async H2D/D2H, CUDA event timing
- Robust SCRFD postproc (stride 8/16/32)
- Optional NVENC (GStreamer) writer
- Pixelation with adjustable strength/margin/max-faces (+ optional noise)

Usage:
ython scrfd_trt_pixelate.py  \ 
    --engine "$HOME/anon-orin/models/scrfd_2.5g_640.engine" \  
    --input "/home/openmind/Desktop/wenjinf-OM-workspace/videos/my_video.mp4" \  
    --out "$HOME/anon-orin/results/out_nvenc_mask.mp4" \  
    --nvenc \  
    --conf 0.5 \
    --topk 100 \
    --max_dets 50 \  
    --pixelate \
    --pixel_blocks 8 \
    --pixel_margin 0.25 
--nvenc->use nvenv or not not include it if using cpu
"""

import argparse
import os
import shlex
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pycuda.driver as cuda  # noqa: F401
except ImportError:
    print("PyCUDA not found, make sure to install it for GPU support.")

sys.path.append("/usr/lib/python3/dist-packages")

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not found, make sure to install it.")

# ---------------- detection / postproc defaults ----------------
CONF_THRES = 0.50
NMS_IOU = 0.40
IGNORE_SHORT = 8
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
TOPK_PER_LEVEL = 150
MAX_DETS = 100
# ---------------------------------------------------------------


def letterbox_bgr(
    img: np.ndarray, new: Tuple[int, int] = (640, 640), color=(114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    Resize with unchanged aspect ratio using padding (letterbox).

    Parameters
    ----------
    img : np.ndarray
        Input BGR image.
    new : Tuple[int, int], optional
        Target (H, W), by default (640, 640).
    color : tuple, optional
        Pad color in BGR, by default (114, 114, 114).

    Returns
    -------
    Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]
        (letterboxed_image, scale_ratio, (left_pad, top_pad), (orig_w, orig_h))
    """
    h, w = img.shape[:2]
    r = min(new[0] / h, new[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_r = (
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if (h, w) != (nh, nw)
        else img
    )
    canvas = np.full((new[0], new[1], 3), color, dtype=np.uint8)
    top = (new[0] - nh) // 2
    left = (new[1] - nw) // 2
    canvas[top : top + nh, left : left + nw] = img_r
    return canvas, r, (left, top), (w, h)


def nms_numpy(dets: Optional[np.ndarray], iou: float = 0.5) -> Optional[np.ndarray]:
    """
    Non-maximum suppression on (x1,y1,x2,y2,score) boxes.

    Parameters
    ----------
    dets : Optional[np.ndarray]
        Detections array (N,5) or None.
    iou : float, optional
        IoU threshold, by default 0.5.

    Returns
    -------
    Optional[np.ndarray]
        Filtered detections (N,5) or None if input is None.
    """
    if dets is None or len(dets) == 0:
        return dets
    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou)[0]
        order = order[inds + 1]
    return dets[keep]


def draw_dets(
    img: np.ndarray,
    dets: np.ndarray,
    color=(0, 255, 0),
    thickness: int = 2,
    put_fps: Optional[str] = None,
    draw_boxes: bool = True,
) -> np.ndarray:
    """
    Draw detection rectangles and optional FPS/latency overlay.

    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in-place).
    dets : np.ndarray
        Detections (N,5): x1,y1,x2,y2,score.
    color : tuple, optional
        Box color in BGR, by default (0, 255, 0).
    thickness : int, optional
        Box line thickness, by default 2.
    put_fps : Optional[str], optional
        Text overlay (e.g., FPS string), by default None.
    draw_boxes : bool, optional
        Toggle drawing boxes, by default True.

    Returns
    -------
    np.ndarray
        The same image (for chaining).
    """
    if draw_boxes and dets is not None:
        for x1, y1, x2, y2, sc in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if put_fps:
        cv2.putText(
            img,
            put_fps,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return img


def make_center_priors(size: int, stride: int, num_anchors: int = 2) -> np.ndarray:
    """
    Create center prior grid (stride-spaced anchor centers).

    Parameters
    ----------
    size : int
        Input size (square).
    stride : int
        Feature map stride.
    num_anchors : int, optional
        Anchors per location, by default 2.

    Returns
    -------
    np.ndarray
        (N,2) center points (cx,cy) repeated for anchors.
    """
    fh = size // stride
    fw = size // stride
    xs = (np.arange(fw) + 0.5) * stride
    ys = (np.arange(fh) + 0.5) * stride
    cx, cy = np.meshgrid(xs, ys)
    pts = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=1)
    return np.repeat(pts, repeats=num_anchors, axis=0).astype(np.float32)


def distance2bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Convert distances (l,t,r,b) to boxes (x1,y1,x2,y2) from center points.

    Parameters
    ----------
    points : np.ndarray
        Centers (N,2).
    distances : np.ndarray
        Distances (N,4) as (l,t,r,b).

    Returns
    -------
    np.ndarray
        Boxes (N,4) as (x1,y1,x2,y2).
    """
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def expand_clip(
    x1: int, y1: int, x2: int, y2: int, margin: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """
    Expand a box by margin and clip to image boundaries.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Original box corners.
    margin : float
        Expansion ratio (e.g., 0.25 = +25%).
    W, H : int
        Image width/height.

    Returns
    -------
    Tuple[int, int, int, int]
        Expanded and clipped box (x1,y1,x2,y2).
    """
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    w *= 1.0 + margin
    h *= 1.0 + margin
    x1n = max(0, int(cx - w * 0.5))
    y1n = max(0, int(cy - h * 0.5))
    x2n = min(W - 1, int(cx + w * 0.5))
    y2n = min(H - 1, int(cy + h * 0.5))
    return x1n, y1n, x2n, y2n


def pixelate_roi(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    blocks_on_short: int = 8,
    noise_sigma: float = 0.0,
) -> None:
    """
    Pixelate an ROI in-place, with optional noise.

    Parameters
    ----------
    img : np.ndarray
        BGR image (modified in-place).
    x1, y1, x2, y2 : int
        ROI corners.
    blocks_on_short : int, optional
        # of blocks along the short side (smaller = stronger), by default 8.
    noise_sigma : float, optional
        Gaussian noise sigma added to pixelated ROI, by default 0.0.

    Returns
    -------
    None
    """
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    short = min(w, h)
    blocks_on_short = max(1, int(blocks_on_short))

    scale = blocks_on_short / float(short)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if noise_sigma and noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, size=big.shape).astype(np.float32)
        big = np.clip(big.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img[y1:y2, x1:x2] = big


def apply_pixelation(
    img: np.ndarray,
    dets: Optional[np.ndarray],
    margin: float = 0.25,
    blocks: int = 8,
    max_faces: int = 32,
    noise_sigma: float = 0.0,
) -> None:
    """
    Pixelate multiple faces by score order up to a maximum.

    Parameters
    ----------
    img : np.ndarray
        Image to modify in-place.
    dets : Optional[np.ndarray]
        Detections (N,5) or None.
    margin : float, optional
        Expansion ratio before pixelation, by default 0.25.
    blocks : int, optional
        Blocks along short side, by default 8.
    max_faces : int, optional
        Max faces to pixelate, by default 32.
    noise_sigma : float, optional
        Gaussian noise sigma, by default 0.0.

    Returns
    -------
    None
    """
    if dets is None or len(dets) == 0:
        return
    H, W = img.shape[:2]
    dets_sorted = dets[dets[:, 4].argsort()[::-1]]
    for x1, y1, x2, y2, _sc in dets_sorted[:max_faces]:
        x1e, y1e, x2e, y2e = expand_clip(
            int(x1), int(y1), int(x2), int(y2), margin, W, H
        )
        pixelate_roi(
            img, x1e, y1e, x2e, y2e, blocks_on_short=blocks, noise_sigma=noise_sigma
        )


class TRTInfer:
    """
    TensorRT wrapper for SCRFD face detection.

    Parameters
    ----------
    engine_path : str
        Path to TensorRT engine (.plan).
    input_name : str, optional
        Engine input tensor name, by default "input.1".
    size : int, optional
        Square input size, by default 640.
    verbose : bool, optional
        Print tensor metadata, by default False.
    """

    def __init__(
        self,
        engine_path: str,
        input_name: str = "input.1",
        size: int = 640,
        verbose: bool = False,
    ):
        self.size = size
        self.verbose = verbose
        import pycuda.driver as cuda

        cuda.init()
        self.stream = cuda.Stream()

        logger = trt.Logger(trt.Logger.WARNING)
        with (
            open(os.path.expanduser(engine_path), "rb") as f,
            trt.Runtime(logger) as rt,
        ):
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.modes = {n: self.engine.get_tensor_mode(n) for n in self.names}
        self.dtypes = {
            n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.names
        }

        self.in_name = input_name
        assert self.in_name in self.names, f"input '{self.in_name}' not in {self.names}"
        if not self.ctx.set_input_shape(self.in_name, (1, 3, self.size, self.size)):
            raise RuntimeError("set_input_shape failed")

        self.shapes = {n: tuple(self.ctx.get_tensor_shape(n)) for n in self.names}
        if self.verbose:
            for n in self.names:
                print(
                    f"[tensor] {n:20s} mode={self.modes[n].name} shape={self.shapes[n]} dtype={self.dtypes[n]}"
                )

        # device alloc + address binding
        import pycuda.driver as cuda2

        self.alloc: Dict[str, Any] = {}
        for n in self.names:
            nbytes = int(np.prod(self.shapes[n])) * np.dtype(self.dtypes[n]).itemsize
            if nbytes == 0:
                continue
            mem = cuda2.mem_alloc(nbytes)
            self.alloc[n] = mem
            self.ctx.set_tensor_address(n, int(mem))

        # pinned host buffers
        self.h_in = cuda2.pagelocked_empty(
            (1, 3, self.size, self.size), dtype=self.dtypes[self.in_name]
        )
        self.out_names = [
            n for n in self.names if self.modes[n] == trt.TensorIOMode.OUTPUT
        ]
        self.h_out = {
            n: cuda2.pagelocked_empty(
                int(np.prod(self.shapes[n])), dtype=self.dtypes[n]
            )
            for n in self.out_names
            if int(np.prod(self.shapes[n])) > 0
        }

        # timing events
        self.ev_start = cuda2.Event()
        self.ev_end = cuda2.Event()

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[float, int, int, int, int]:
        """
        Letterbox + normalize + NCHW convert into pinned input buffer.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input BGR frame.

        Returns
        -------
        Tuple[float, int, int, int, int]
            (scale_ratio, left_pad, top_pad, orig_W, orig_H)
        """
        inp, r, (left, top), (W, H) = letterbox_bgr(frame_bgr, (self.size, self.size))
        x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1)).astype(self.h_in.dtype, copy=False)
        self.h_in[...] = x[None]
        return r, left, top, W, H

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on one frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input BGR frame.

        Returns
        -------
        Tuple[np.ndarray, float]
            (dets, gpu_ms) where dets is (N,5) and gpu_ms is kernel time in ms.
        """
        import pycuda.driver as cuda3

        r, left, top, W, H = self._preprocess(frame_bgr)
        cuda3.memcpy_htod_async(self.alloc[self.in_name], self.h_in, self.stream)

        self.ev_start.record(self.stream)
        if not self.ctx.execute_async_v3(self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.ev_end.record(self.stream)

        for n in self.out_names:
            if n in self.alloc and n in self.h_out:
                cuda3.memcpy_dtoh_async(self.h_out[n], self.alloc[n], self.stream)

        self.stream.synchronize()
        gpu_ms = self.ev_end.time_since(self.ev_start)

        outs: Dict[str, np.ndarray] = {}
        for n in self.out_names:
            if n in self.h_out:
                outs[n] = np.array(self.h_out[n]).reshape(self.shapes[n])

        dets = self._postproc_scrfd(outs, left, top, r, W, H)
        return dets, gpu_ms

    def _postproc_scrfd(
        self, outs: Dict[str, np.ndarray], left: int, top: int, r: float, W: int, H: int
    ) -> np.ndarray:
        """
        Convert SCRFD raw outputs to filtered XYXY detections.

        Parameters
        ----------
        outs : Dict[str, np.ndarray]
            Engine output tensors.
        left, top : int
            Letterbox paddings.
        r : float
            Letterbox scale ratio.
        W, H : int
            Original image width/height.

        Returns
        -------
        np.ndarray
            Detections (N,5) as (x1,y1,x2,y2,score).
        """
        scores_map = {
            arr.shape[0]: arr.reshape(-1)
            for arr in outs.values()
            if arr.ndim == 2 and arr.shape[-1] == 1
        }
        bboxes_map = {
            arr.shape[0]: arr.reshape(-1, 4)
            for arr in outs.values()
            if arr.ndim == 2 and arr.shape[-1] == 4
        }
        if not scores_map or not bboxes_map:
            return np.zeros((0, 5), dtype=np.float32)

        all_scores, all_boxes = [], []
        for stride in STRIDES:
            fh = self.size // stride
            fw = self.size // stride
            length = fh * fw * NUM_ANCHORS
            if length not in scores_map or length not in bboxes_map:
                continue
            sc = scores_map[length]
            bb = bboxes_map[length]

            # If logits, convert to probs
            if not (0.0 <= sc.min() and sc.max() <= 1.0):
                sc = 1.0 / (1.0 + np.exp(-sc))

            A = NUM_ANCHORS
            sc2 = sc.reshape(fh * fw, A)
            bb2 = bb.reshape(fh * fw, A, 4)
            idx = np.argmax(sc2, axis=1)
            sc1 = sc2[np.arange(fh * fw), idx]
            bb1 = bb2[np.arange(fh * fw), idx, :]

            # stride-multiplication heuristic
            if np.median(bb1) < 12.0:
                bb1 = bb1 * float(stride)

            pri = make_center_priors(self.size, stride, 1)
            boxes_xyxy = distance2bbox(pri, bb1.astype(np.float32))

            keep = sc1 >= getattr(self, "conf_thres", CONF_THRES)
            if not np.any(keep):
                continue
            sc1 = sc1[keep]
            boxes_xyxy = boxes_xyxy[keep]

            topk = int(getattr(self, "topk_per_level", TOPK_PER_LEVEL))
            if sc1.size > topk:
                idk = np.argpartition(sc1, -topk)[-topk:]
                sc1 = sc1[idk]
                boxes_xyxy = boxes_xyxy[idk]

            all_scores.append(sc1)
            all_boxes.append(boxes_xyxy)

        if not all_scores:
            return np.zeros((0, 5), dtype=np.float32)

        scores = np.concatenate(all_scores, axis=0)
        boxes = np.concatenate(all_boxes, axis=0)

        gtopk = max(
            int(getattr(self, "topk_per_level", TOPK_PER_LEVEL)) * len(STRIDES),
            int(getattr(self, "max_dets", MAX_DETS)) * 4,
        )
        if scores.size > gtopk:
            idg = np.argpartition(scores, -gtopk)[-gtopk:]
            scores = scores[idg]
            boxes = boxes[idg]

        dets = []
        for b, s in zip(boxes, scores):
            x1 = (b[0] - left) / r
            y1 = (b[1] - top) / r
            x2 = (b[2] - left) / r
            y2 = (b[3] - top) / r
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W - 1, x2)))
            y2 = int(max(0, min(H - 1, y2)))
            if min(x2 - x1, y2 - y1) < IGNORE_SHORT:
                continue
            dets.append([x1, y1, x2, y2, float(s)])

        if not dets:
            return np.zeros((0, 5), dtype=np.float32)
        dets = np.array(dets, dtype=np.float32)
        dets = nms_numpy(dets, iou=NMS_IOU)

        K = int(getattr(self, "max_dets", MAX_DETS))
        if dets.shape[0] > K:
            dets = dets[dets[:, 4].argsort()[::-1][:K]]
        return dets


def open_nvenc_writer(
    out_path: str, width: int, height: int, fps_in: float
) -> Optional[cv2.VideoWriter]:
    """
    Create a GStreamer NVENC writer for OpenCV.

    Parameters
    ----------
    out_path : str
        Output file path (.mp4 or .mkv fallback).
    width : int
        Frame width.
    height : int
        Frame height.
    fps_in : float
        Input FPS (used to set pipeline framerate).

    Returns
    -------
    Optional[cv2.VideoWriter]
        Opened VideoWriter or None if creation failed.
    """
    fps = float(fps_in) if fps_in and fps_in > 0 else 25.0
    fps_i = int(round(fps))

    candidates = [
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! queue leaky=downstream max-size-buffers=1 "
            f"! videoconvert "
            f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 "
            f"! h264parse ! mp4mux "
            f"! filesink location={out_path} sync=false"
        ),
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! videoconvert "
            f"! nvv4l2h264enc insert-sps-pps=true "
            f"! h264parse ! mp4mux "
            f"! filesink location={out_path} sync=false"
        ),
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! videoconvert "
            f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true "
            f"! h264parse ! matroskamux "
            f"! filesink location={os.path.splitext(out_path)[0]}.mkv sync=false"
        ),
    ]

    for p in candidates:
        w = cv2.VideoWriter(p, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        if w.isOpened():
            print("[nvenc] using pipeline:", p)
            return w
    return None


def build_gst_filesrc_pipeline(path: str) -> str:
    """
    Build a Jetson-friendly GStreamer read pipeline for a given file path.

    Parameters
    ----------
    path : str
        Input media path.

    Returns
    -------
    str
        GStreamer pipeline string ending with appsink.
    """
    # Quote the path to be safe with spaces/special chars
    loc = shlex.quote(os.path.expanduser(path))
    return (
        f"filesrc location={loc} ! qtdemux ! h264parse ! nvv4l2h264dec enable-max-performance=1 "
        f"! nvvidconv ! video/x-raw,format=BGRx ! videoconvert "
        f"! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )


def open_input_capture(user_input: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    """
    Open the input using GStreamer if possible, otherwise fallback to plain OpenCV.

    Behavior
    --------
    - If `user_input` already looks like a GStreamer pipeline (contains '!'), try CAP_GSTREAMER.
    - If `user_input` is a file path, first try a generated filesrc→decoder→BGR appsink
      pipeline via CAP_GSTREAMER. If that fails, fallback to cv2.VideoCapture(path).

    Parameters
    ----------
    user_input : str
        File path or GStreamer pipeline.

    Returns
    -------
    Tuple[cv2.VideoCapture, int, int, float]
        (cap, width, height, fps)
    """
    inp = user_input.strip()
    cap = None

    # Case 1: treat explicit pipeline
    is_pipeline = "!" in inp
    if is_pipeline:
        cap = cv2.VideoCapture(inp, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open GStreamer pipeline: {inp}")
    else:
        # Case 2: path → try GStreamer filesrc pipeline first
        gst = build_gst_filesrc_pipeline(inp)
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            # Fallback to plain OpenCV reader
            cap = cv2.VideoCapture(inp)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open input: {inp}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cap, W, H, fps_in


def main() -> None:
    """
    Entry point: run SCRFD TensorRT inference on an input video stream and
    optionally apply pixelation + NVENC output.
    """
    ap = argparse.ArgumentParser()

    # Detection knobs
    ap.add_argument(
        "--conf", type=float, default=None, help="Confidence threshold override"
    )
    ap.add_argument("--topk", type=int, default=None, help="Top-K per level override")
    ap.add_argument(
        "--max_dets", type=int, default=None, help="Global max detections override"
    )

    # IO / engine
    ap.add_argument("--engine", required=True, help="Path to TensorRT engine (.plan)")
    ap.add_argument(
        "--input", required=True, help="Input file path or GStreamer pipeline"
    )
    ap.add_argument("--out", default="", help="Optional output file path")
    ap.add_argument("--size", type=int, default=640, help="Model input size (square)")
    ap.add_argument("--input_name", default="input.1", help="Engine input tensor name")
    ap.add_argument("--print_every", type=int, default=10, help="Log interval (frames)")
    ap.add_argument(
        "--nvenc", action="store_true", help="Use NVENC via GStreamer for output"
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose tensor metadata")

    # Pixelation knobs
    ap.add_argument(
        "--pixelate", action="store_true", help="Enable pixelation anonymization"
    )
    ap.add_argument(
        "--pixel_blocks",
        type=int,
        default=8,
        help="Short-side blocks (smaller = stronger)",
    )
    ap.add_argument(
        "--pixel_margin", type=float, default=0.25, help="Expand bbox by this ratio"
    )
    ap.add_argument(
        "--pixel_max_faces", type=int, default=32, help="Max faces pixelated per frame"
    )
    ap.add_argument(
        "--pixel_noise",
        type=float,
        default=0.0,
        help="Gaussian noise sigma after pixelation",
    )
    ap.add_argument("--no_boxes", action="store_true", help="Do not draw rectangles")

    args = ap.parse_args()

    eng = os.path.expanduser(args.engine)
    assert os.path.exists(eng), f"engine not found: {eng}"

    # Open input (path → try GStreamer first; fallback to plain OpenCV)
    cap, W, H, fps_in = open_input_capture(args.input)

    # Output writer
    writer = None
    if args.out:
        out_path = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if args.nvenc:
            w = open_nvenc_writer(out_path, W, H, fps_in)
            if w is not None and w.isOpened():
                writer = w
            else:
                print("[warn] NVENC pipeline failed to open, falling back to CPU mp4v")
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, fps_in if fps_in > 0 else 25.0, (W, H)
            )
            if not writer.isOpened():
                raise RuntimeError("CPU VideoWriter failed to open")

    infer = TRTInfer(
        engine_path=eng,
        input_name=args.input_name,
        size=args.size,
        verbose=args.verbose,
    )
    if args.conf is not None:
        infer.conf_thres = args.conf
    if args.topk is not None:
        infer.topk_per_level = args.topk
    if args.max_dets is not None:
        infer.max_dets = args.max_dets

    # Warmup
    for _ in range(8):
        ok, f = cap.read()
        if not ok:
            break
        _ = infer.infer(f)

    total_frames = 0
    t0 = time.perf_counter()
    ema_ms = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets, gpu_ms = infer.infer(frame)
        total_frames += 1
        ema_ms = gpu_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * gpu_ms)

        # Pixelation
        if args.pixelate and dets is not None and len(dets) > 0:
            apply_pixelation(
                frame,
                dets,
                margin=args.pixel_margin,
                blocks=args.pixel_blocks,
                max_faces=args.pixel_max_faces,
                noise_sigma=args.pixel_noise,
            )

        # Overlay & write
        if writer is not None:
            elapsed = time.perf_counter() - t0
            fps_now = total_frames / elapsed if elapsed > 0 else 0.0
            overlay = f"GPU {gpu_ms:.2f} ms | EMA {ema_ms:.2f} ms | FPS {fps_now:.1f} | faces {len(dets)}"
            out_frame = draw_dets(
                frame, dets, put_fps=overlay, draw_boxes=(not args.no_boxes)
            )
            writer.write(out_frame)

        if total_frames % args.print_every == 0:
            elapsed = time.perf_counter() - t0
            fps_now = total_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"[{total_frames:05d}] GPU={gpu_ms:.2f} ms  EMA={ema_ms:.2f} ms  FPS={fps_now:.1f}  faces={len(dets)}"
            )

    cap.release()
    if writer is not None:
        writer.release()

    if total_frames:
        total_time = time.perf_counter() - t0
        print(
            f"done. frames={total_frames} avg_fps={total_frames / total_time:.2f} avg_gpu_ms≈{ema_ms:.2f}"
        )


if __name__ == "__main__":
    main()
