import asyncio
import logging
import os
import sqlite3
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import yaml

from .io_provider import IOProvider
from .singleton import singleton

logger = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_image_webp(path: str, bgr_img: np.ndarray, q: int = 90) -> None:
    params = [int(cv2.IMWRITE_WEBP_QUALITY), int(q)]
    if not cv2.imwrite(path, bgr_img, params):
        raise RuntimeError(f"Failed to write {path}")


def variance_of_laplacian(image_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def crop_from_bbox(img: np.ndarray, xyxy, expand: float = 0.0) -> Optional[np.ndarray]:
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


def image_downscale_long(bgr: np.ndarray, long_side: int = 1280) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return bgr
    scale = long_side / float(m)
    return cv2.resize(
        bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def phash64(bgr: np.ndarray) -> str:
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


def hamming64_hex(a_hex: str, b_hex: str) -> int:
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    return (a ^ b).bit_count()


DEFAULT_CFG = {
    "storage": {
        "crops_dir": "lost_and_found_data/crops",
        "frames_dir": "lost_and_found_data/images",
        "sqlite_path": "lost_and_found_data/store.sqlite",
        "vector_index": "lost_and_found_data/index.ann",
        "save_topk_per_frame": 6,
        "frame_long_side": 1280,
        "webp_quality_frames": 80,
        "similar_hamming_threshold": 8,
        "max_frames": 5000,
        "max_disk_gb": 5.0,
    },
    "metadata": {
        "default_room": "living_room",
        "room_dynamic_key": "room_type",
    },
}

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

    def inc_ref(self, frame_id: int, delta: int) -> None:
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
        Delete all sightings for a label except the given sighting_id, and
        clean up associated crops/scenes/frames.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, frame_id, crop_path, scene_path
            FROM sightings
            WHERE label = ? AND id != ?
            """,
            (label, keep_sighting_id),
        )
        rows = cur.fetchall()
        if not rows:
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

        cur.execute(
            "DELETE FROM sightings WHERE label = ? AND id != ?",
            (label, keep_sighting_id),
        )
        self.conn.commit()


@singleton
class LostAndFoundIngestProvider:
    """
    Ingest provider:

    - Polls IOProvider for latest detection packet published by YoloDetectRTSPProvider
      under key 'yolo_latest'.
    - For each new packet:
        * saves frame (with pHash-based dedupe),
        * saves crops & scenes,
        * inserts sightings into SQLite,
        * keeps only the latest sighting per label.
    """

    IO_KEY_LATEST = "yolo_latest"

    def __init__(
        self,
        cfg_path: str | None = None,
        poll_interval: float = 0.3,
    ):
        self.running: bool = False
        self.poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self._last_processed_ts: int = 0

        # Load config
        if cfg_path is not None and os.path.exists(cfg_path):
            try:
                self.cfg = yaml.safe_load(open(cfg_path, "r"))
            except Exception:
                logger.exception("Failed to load ingest cfg_path; using defaults")
                self.cfg = DEFAULT_CFG.copy()
        else:
            self.cfg = DEFAULT_CFG.copy()

        storage_cfg = self.cfg["storage"]
        meta_cfg = self.cfg.get("metadata", {})

        self.store = Store(storage_cfg["sqlite_path"])

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

    # -------------------- Room lookup --------------------

    def _get_room(self) -> str:
        try:
            val = self.io_provider.get_dynamic_variable(self.room_dynamic_key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except AttributeError:
            pass
        except Exception:
            logger.exception("Failed to read room from IOProvider; using default_room")
        return self.default_room

    # -------------------- External API --------------------

    def start(self) -> None:
        if self.running:
            logger.warning("LostAndFoundIngestProvider is already running")
            return
        self.running = True
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._loop())
        logger.info("LostAndFoundIngestProvider started")

    def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("LostAndFoundIngestProvider stopped")

    # -------------------- Main loop --------------------

    async def _loop(self) -> None:
        try:
            while True:
                packet = self.io_provider.get_dynamic_variable(self.IO_KEY_LATEST)
                if (
                    packet
                    and isinstance(packet, dict)
                    and packet.get("ts", 0) > self._last_processed_ts
                ):
                    self._last_processed_ts = int(packet["ts"])
                    await asyncio.to_thread(self._process_packet_sync, packet)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[Ingest] loop error")

    # -------------------- Heavy processing --------------------

    def _process_packet_sync(self, packet: dict[str, Any]) -> None:
        frame = packet.get("frame")
        dets = packet.get("detections") or []
        ts = int(packet.get("ts", now_ms()))

        if frame is None or not isinstance(frame, np.ndarray):
            logger.debug("[Ingest] packet without valid frame; skipping")
            return

        if not dets:
            logger.debug(f"[Ingest] no detections for ts={ts}; skipping")
            return

        storage_cfg = self.cfg["storage"]

        # Save downscaled full frame (with pHash dedupe)
        frame_long_side = storage_cfg.get("frame_long_side", 1280)
        frame_ds = image_downscale_long(frame, frame_long_side)
        frame_phash = phash64(frame_ds)
        prev = self.store.get_last_frame()

        keep_frame = True
        if prev:
            hd = hamming64_hex(prev["phash"], frame_phash)
            thr = storage_cfg.get("similar_hamming_threshold", 8)
            if hd <= thr:
                keep_frame = True

        frame_path = ""
        frame_id: Optional[int] = None
        if keep_frame:
            frame_path = os.path.join(self.frames_dir, f"{ts}.webp")
            save_image_webp(
                frame_path,
                frame_ds,
                q=storage_cfg.get("webp_quality_frames", 80),
            )
            hds, wds = frame_ds.shape[:2]
            frame_id = self.store.insert_frame(
                ts=ts,
                path=frame_path,
                phash=frame_phash,
                wh=(wds, hds),
            )

        # Sort detections and keep top-k
        dets_sorted = sorted(dets, key=lambda d: d["conf"], reverse=True)[
            : self.save_topk
        ]
        logging.debug(f"[Ingest] processing {len(dets_sorted)} detections for ts={ts}")

        something_referenced = False
        for d in dets_sorted:
            xyxy = d["xyxy"]
            label = d["label"]
            conf = float(d["conf"])

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
                f"[Ingest] Detected '{label}' (conf={conf:.2f}, sharp={sharp:.1f}) "
                f"in room '{room}' at ts={ts}. crop={crop_path}"
            )

            # Update "latest sighting" mapping for this label
            self.store.set_latest_for_label(
                label,
                sighting_id=sighting_id,
                frame_id=frame_id if frame_id is not None else -1,
            )

            # Ensure we only keep the latest sighting for this label:
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
