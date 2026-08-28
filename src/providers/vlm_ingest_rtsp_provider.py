import asyncio
import base64
import json
import logging
import os
import sqlite3
import time
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import yaml
from om1_vlm import VideoRTSPStream
from ultralytics import YOLO

from .io_provider import IOProvider
from .singleton import singleton


def now_ms():
    return int(time.time() * 1000)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_image_webp(path, bgr_img, q=90):
    params = [int(cv2.IMWRITE_WEBP_QUALITY), int(q)]
    if not cv2.imwrite(path, bgr_img, params):
        raise RuntimeError(f"Failed to write {path}")


def variance_of_laplacian(image_gray):
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def crop_from_bbox(img, xyxy, expand=0.0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    if expand > 0:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1), (y2 - y1)
        bw *= 1 + expand
        bh *= 1 + expand
        x1, x2 = int(cx - bw / 2), int(cx + bw / 2)
        y1, y2 = int(cy - bh / 2), int(cy + bh / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def image_downscale_long(bgr, long_side=1280):
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return bgr
    scale = long_side / float(m)
    return cv2.resize(
        bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def phash64(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    gray32 = gray.astype(np.float32)
    dct = cv2.dct(gray32)
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bits = (dct_low > med).astype(np.uint8).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def hamming64_hex(a_hex, b_hex):
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    return (a ^ b).bit_count()


logger = logging.getLogger(__name__)

DEFAULT_CFG = {
    "camera": {
        "src": 0,
        "fps": 10,
        "save_every_n": 10,
    },
    "detect": {
        "model": "yolov8n.pt",
        "conf": 0.35,
        "img_size": 640,
        "blacklist": ["person"],
        "whitelist": ["tv", "cell phone", "cup", "bottle", "book"],
    },
    "storage": {
        "crops_dir": "lost_and_found_data/crops",
        "frames_dir": "lost_and_found_data/images",
        "sqlite_path": "lost_and_found_db/store.sqlite",
        "vector_index": "lost_and_found_db/index.ann",
        "save_topk_per_frame": 6,
        "frame_long_side": 1280,
        "webp_quality_frames": 80,
        "similar_hamming_threshold": 8,
        "max_frames": 5000,
        "max_disk_gb": 5.0,
    },
    "metadata": {
        "default_room": "living_room",
        # key from IOProvider (matches room_type variable)
        "room_dynamic_key": "room_type",
    },
}

# --------------------------------------------------------------------------------------
# SQLite schema + store
# --------------------------------------------------------------------------------------

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sightings(
        id INTEGER PRIMARY KEY,
        ts INTEGER,
        room TEXT,
        label TEXT,
        conf REAL,
        frame_path TEXT,
        crop_path TEXT,
        x1 INT, y1 INT, x2 INT, y2 INT,
        w INT, h INT,
        sharpness REAL,
        scene_path TEXT,
        frame_id INT
    );""",
    """CREATE TABLE IF NOT EXISTS frames(
        id INTEGER PRIMARY KEY,
        ts INTEGER,
        path TEXT,
        phash TEXT,
        w INT, h INT,
        refcount INT DEFAULT 0
    );""",
    """CREATE TABLE IF NOT EXISTS latest_by_label(
        label TEXT PRIMARY KEY,
        sighting_id INT,
        frame_id INT
    );""",
    "CREATE INDEX IF NOT EXISTS idx_lbl_ts ON sightings(label, ts);",
    "CREATE INDEX IF NOT EXISTS idx_room_ts ON sightings(room, ts);",
    "CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts);",
]


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        for stmt in SCHEMA:
            self.conn.execute(stmt)
        self.conn.commit()

    # -------- frames --------

    def insert_frame(self, ts: int, path: str, phash: str, wh: Tuple[int, int]) -> int:
        w, h = wh
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO frames(ts, path, phash, w, h, refcount) "
            "VALUES (?,?,?,?,?,0)",
            (ts, path, phash, w, h),
        )
        self.conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("insert_frame: lastrowid is None after INSERT")
        return int(rowid)

    def inc_ref(self, frame_id: int, delta: int):
        self.conn.execute(
            "UPDATE frames SET refcount = refcount + ? WHERE id=?",
            (delta, frame_id),
        )
        self.conn.commit()

    def get_last_frame(self):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, ts, path, phash, w, h, refcount "
            "FROM frames ORDER BY id DESC LIMIT 1"
        )
        r = cur.fetchone()
        if not r:
            return None
        k = ["id", "ts", "path", "phash", "w", "h", "refcount"]
        return dict(zip(k, r))

    def get_frame_meta(self, frame_id: int):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, ts, path, phash, w, h, refcount FROM frames WHERE id=?",
            (frame_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        k = ["id", "ts", "path", "phash", "w", "h", "refcount"]
        return dict(zip(k, r))

    def delete_frame_if_unref(self, frame_id: int) -> bool:
        m = self.get_frame_meta(frame_id)
        if not m:
            return False
        if m["refcount"] <= 0 and m["path"] and os.path.exists(m["path"]):
            try:
                os.remove(m["path"])
            except Exception:
                pass
            self.conn.execute("DELETE FROM frames WHERE id=?", (frame_id,))
            self.conn.commit()
            return True
        return False

    # -------- sightings --------

    def insert_sighting(
        self,
        ts: int,
        room: str,
        label: str,
        conf: float,
        frame_path: str,
        crop_path: str,
        box: Tuple[int, int, int, int],
        wh: Tuple[int, int],
        sharpness: float,
        scene_path: str = "",
        frame_id: int | None = None,
    ) -> int:
        x1, y1, x2, y2 = box
        w, h = wh
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO sightings(
                ts, room, label, conf, frame_path, crop_path,
                x1, y1, x2, y2, w, h, sharpness, scene_path, frame_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                room,
                label,
                conf,
                frame_path,
                crop_path,
                x1,
                y1,
                x2,
                y2,
                w,
                h,
                sharpness,
                scene_path,
                frame_id,
            ),
        )
        self.conn.commit()
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("insert_sighting: lastrowid is None after INSERT")
        return int(rowid)

    # -------- latest_by_label --------

    def get_latest_for_label(self, label: str):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT label, sighting_id, frame_id FROM latest_by_label WHERE label=?",
            (label,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return dict(label=r[0], sighting_id=r[1], frame_id=r[2])

    def set_latest_for_label(self, label: str, sighting_id: int, frame_id: int | None):
        self.conn.execute(
            "REPLACE INTO latest_by_label(label, sighting_id, frame_id) "
            "VALUES (?, ?, ?)",
            (label, sighting_id, frame_id if frame_id is not None else -1),
        )
        self.conn.commit()

    def delete_other_sightings_for_label(
        self, label: str, keep_sighting_id: int
    ) -> None:
        """
        Delete all sightings for a label except the given sighting_id.

        Also:
          - decrements frame refcounts for the deleted sightings
          - removes any frames that become unreferenced
          - deletes crop and scene image files for the deleted sightings
        """
        cur = self.conn.cursor()
        # Fetch all other sightings for this label, including file paths
        cur.execute(
            """
            SELECT id, frame_id, crop_path, scene_path
            FROM sightings
            WHERE label = ? AND id != ?
            """,
            (label, keep_sighting_id),
        )
        rows = cur.fetchall()
        if rows is None:
            return

        for sid, frame_id, crop_path, scene_path in rows:
            # Delete crop file
            if crop_path and os.path.exists(crop_path):
                try:
                    os.remove(crop_path)
                except Exception:
                    pass

            # Delete scene file
            if scene_path and os.path.exists(scene_path):
                try:
                    os.remove(scene_path)
                except Exception:
                    pass

            # Adjust frame refcount and maybe delete frame file+row
            if frame_id is not None and frame_id != -1:
                self.inc_ref(frame_id, -1)
                self.delete_frame_if_unref(frame_id)

        # Delete the old sighting rows themselves
        cur.execute(
            "DELETE FROM sightings WHERE label = ? AND id != ?",
            (label, keep_sighting_id),
        )
        self.conn.commit()


# --------------------------------------------------------------------------------------
# YOLO detector
# --------------------------------------------------------------------------------------


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        img_size: int = 640,
        blacklist=None,
        whitelist=None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.img_size = img_size
        self.blacklist = set(blacklist or [])
        self.whitelist = set(whitelist or [])

    def infer(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        res = self.model.predict(
            source=frame_bgr, imgsz=self.img_size, conf=self.conf, verbose=False
        )[0]
        out: list[dict[str, Any]] = []
        boxes: Any = getattr(res, "boxes", None)
        if boxes is None:
            return out
        for b in boxes:
            cls_id = int(b.cls.item())
            label = res.names[cls_id]
            if label in self.blacklist:
                continue
            if self.whitelist and (label not in self.whitelist):
                continue
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf.item())
            out.append(dict(xyxy=xyxy, label=label, conf=conf))
        return out


# --------------------------------------------------------------------------------------
# Simple Lost & Found ingest provider
# --------------------------------------------------------------------------------------


@singleton
class VLMIngestRTSPProvider:
    """
    Simple Lost & Found ingest provider:

      - Subscribes to VideoRTSPStream via a frame callback (base64 JSON)
      - Processes frames on a background queue with stride + drop-oldest
      - Runs YOLO detection and stores:
          * frames in SQLite (frames table)
          * sightings in SQLite (sightings table)
          * the latest sighting per YOLO label in latest_by_label
      - Uses `room_type` from IOProvider if available, otherwise `default_room` from config.
    """

    def __init__(
        self,
        cfg_path: str | None = None,  # no default config file required
        rtsp_url: str = "rtsp://localhost:8554/top_camera",
        decode_format: str = "H264",
        fps: int = 10,
        ingest_stride: int = 10,  # process every Nth frame
        queue_max: int = 10,  # max queued frames before dropping oldest
    ):

        self.running: bool = False

        # Load config & components
        if cfg_path is not None and os.path.exists(cfg_path):
            self.cfg = yaml.safe_load(open(cfg_path, "r"))
        else:
            # fall back to built-in defaults (no external config file needed)
            self.cfg = DEFAULT_CFG.copy()

        storage_cfg = self.cfg["storage"]
        detect_cfg = self.cfg["detect"]
        meta_cfg = self.cfg.get("metadata", {})

        self.store = Store(storage_cfg["sqlite_path"])
        self.det = Detector(
            detect_cfg["model"],
            detect_cfg["conf"],
            detect_cfg["img_size"],
            detect_cfg.get("blacklist"),
            detect_cfg.get("whitelist"),
        )

        # Room: dynamic via IOProvider, fallback to default_room
        self.io_provider = IOProvider()
        self.default_room = meta_cfg.get("default_room", "unknown")
        self.room_dynamic_key = meta_cfg.get("room_dynamic_key", "room_type")

        # How many detections per frame we keep
        self.save_topk = int(storage_cfg.get("save_topk_per_frame", 6))

        # IO paths
        self.frames_dir = storage_cfg["frames_dir"]
        self.crops_dir = storage_cfg["crops_dir"]
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

        # Stream
        self.video_stream: VideoRTSPStream = VideoRTSPStream(
            rtsp_url,
            decode_format,
            frame_callback=self._on_frame,
            fps=fps,
        )
        self._owns_stream = True

        # Ingest control
        self._ingest_stride = max(1, int(ingest_stride))
        self._frame_counter = 0
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(queue_max)))

    # -------------------- Room lookup --------------------

    def _get_room(self) -> str:
        """
        Get the current room name for sightings.

        Prefers IOProvider dynamic variable (room_dynamic_key), falls back to default_room.
        """
        try:
            val = self.io_provider.get_dynamic_variable(self.room_dynamic_key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except AttributeError:
            # IOProvider doesn't have get_dynamic_variable -> just ignore
            pass
        except Exception:
            logger.exception("Failed to read room from IOProvider; using default_room")

        return self.default_room

    # -------------------- External API --------------------

    def register_frame_callback(self, video_callback: Optional[Callable]):
        if video_callback is not None:
            self.video_stream.register_frame_callback(video_callback)

    def start(self):
        if self.running:
            logger.warning("VLMIngestRTSPProvider is already running")
            return
        self.running = True

        if self._owns_stream:
            self.video_stream.start()

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._consumer())
        logger.info("Simple Lost&Found ingest RTSP provider started")

    def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()

        if self._owns_stream:
            try:
                self.video_stream.stop()
            except Exception:
                pass
        logger.info("Simple Lost&Found ingest RTSP provider stopped")

    # -------------------- Callbacks & Worker --------------------

    def _on_frame(self, frame_data: str):
        """
        Receives base64 JSON from VideoRTSPStream, applies stride, and enqueues (drop-oldest).
        frame_data: JSON string {"timestamp": float, "frame": <base64 jpeg>}
        """
        try:
            self._frame_counter += 1
            if (self._frame_counter % self._ingest_stride) != 0 or not self.running:
                return

            d = json.loads(frame_data)
            b = base64.b64decode(d["frame"])
            arr = np.frombuffer(b, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            payload = {"timestamp": d.get("timestamp", time.time()), "frame": frame}

            if self._queue.full():
                try:
                    self._queue.get_nowait()  # drop oldest
                except Exception:
                    pass
            self._queue.put_nowait(payload)

        except Exception as e:
            logger.warning(f"Ingest enqueue error: {e}")

    def _process_item_sync(self, item):
        """
        Synchronous heavy processing for a single queued frame:
        - frame saving / hashing
        - YOLO detection
        - crops + scene saves
        - DB writes

        This runs off the asyncio event loop via asyncio.to_thread.
        """
        frame = item["frame"]
        ts = now_ms()

        # Save downscaled full frame (with pHash dedupe)
        frame_long_side = self.cfg["storage"].get("frame_long_side", 1280)
        frame_ds = image_downscale_long(frame, frame_long_side)
        frame_phash = phash64(frame_ds)
        prev = self.store.get_last_frame()

        keep_frame = True
        if prev:
            hd = hamming64_hex(prev["phash"], frame_phash)
            thr = self.cfg["storage"].get("similar_hamming_threshold", 8)
            if hd <= thr:
                keep_frame = True

        frame_path = ""
        frame_id = None
        if keep_frame:
            frame_path = os.path.join(self.frames_dir, f"{ts}.webp")
            save_image_webp(
                frame_path,
                frame_ds,
                q=self.cfg["storage"].get("webp_quality_frames", 80),
            )
            hds, wds = frame_ds.shape[:2]
            frame_id = self.store.insert_frame(
                ts=ts,
                path=frame_path,
                phash=frame_phash,
                wh=(wds, hds),
            )

        # YOLO detect (sorted by conf, keep top-k)
        dets = self.det.infer(frame)
        logging.debug(
            f"[Lost&Found] YOLO returned {len(dets)} raw detections for frame ts={ts}"
        )
        dets = sorted(dets, key=lambda d: d["conf"], reverse=True)[: self.save_topk]

        something_referenced = False
        for d in dets:
            xyxy = d["xyxy"]
            label = d["label"]
            conf = d["conf"]

            crop = crop_from_bbox(frame, xyxy, expand=0.05)
            if crop is None:
                continue
            h, w = crop.shape[:2]
            if min(h, w) < 64:
                continue

            # Sharpness filter
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = variance_of_laplacian(gray)
            if sharp < 20:
                continue

            # Scene crop (wider area around object)
            scene = crop_from_bbox(frame, xyxy, expand=2.5)
            scene_path = ""
            if scene is not None:
                scene_path = os.path.join(self.frames_dir, f"{ts}_scene.webp")
                save_image_webp(scene_path, scene, q=90)

            crop_path = os.path.join(
                self.crops_dir,
                f"{ts}_{label}_{int(1000 * conf)}.webp",
            )
            save_image_webp(crop_path, crop, q=90)

            if frame_id is not None:
                self.store.inc_ref(frame_id, +1)
                something_referenced = True

            x1, y1, x2, y2 = map(int, xyxy)
            room = self._get_room()

            sighting_id = self.store.insert_sighting(
                ts,
                room,
                label,
                conf,
                frame_path,
                crop_path,
                (x1, y1, x2, y2),
                (w, h),
                float(sharp),
                scene_path=scene_path,
                frame_id=frame_id,
            )

            logging.info(
                f"[Lost&Found] Detected '{label}' (conf={conf:.2f}, sharp={sharp:.1f}) "
                f"in room '{room}' at ts={ts}. crop={crop_path}"
            )

            # Update "latest sighting" mapping for this label
            self.store.set_latest_for_label(
                label,
                sighting_id=sighting_id,
                frame_id=frame_id if frame_id is not None else -1,
            )

            # Ensure we only keep the latest sighting for this label:
            # delete all older sightings
            self.store.delete_other_sightings_for_label(
                label=label,
                keep_sighting_id=sighting_id,
            )

        # drop new frame if nothing referenced it
        if frame_id is not None and not something_referenced:
            self.store.delete_frame_if_unref(frame_id)

        # drop previous similar (unref) frame
        if prev and keep_frame:
            self.store.delete_frame_if_unref(prev["id"])

    async def _consumer(self):
        """
        Background worker that pulls frames and offloads heavy processing
        to a separate thread so the asyncio event loop stays responsive.
        """
        try:
            while True:
                item = await self._queue.get()
                # Offload heavy work (YOLO, I/O, DB) to a thread
                await asyncio.to_thread(self._process_item_sync, item)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Ingest worker error: {e}")
