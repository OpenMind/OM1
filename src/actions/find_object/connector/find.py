import logging
import sqlite3
from contextlib import closing
from typing import Any, Dict, List, Optional

from actions.base import ActionConfig, ActionConnector
from actions.find_object.interface import LostAndFoundQueryInput


class LostAndFoundConnector(ActionConnector[LostAndFoundQueryInput]):
    """
    Connector that queries the Lost & Found SQLite database and logs
    objects that have been detected.

    It reads from the same database that the Lost & Found ingest provider writes to:
        lost_and_found_db/store.sqlite
    unless overridden via ActionConfig.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the LostAndFoundConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.

        Supported config attributes:
        - sqlite_path : str (default "lost_and_found_db/store.sqlite")
        - default_limit : int (default 5)
        """
        super().__init__(config)

        self.sqlite_path: str = getattr(
            config, "sqlite_path", "lost_and_found_db/store.sqlite"
        )
        self.default_limit: int = int(getattr(config, "default_limit", 5))

        if not self.sqlite_path:
            logging.error("LostAndFound connector missing 'sqlite_path' in config")

    async def connect(self, input_protocol: LostAndFoundQueryInput) -> None:
        """
        Handle Lost & Found queries.

        Parameters
        ----------
        input_protocol : LostAndFoundQueryInput
            The input protocol containing:
              - action: label or "all"
              - room: optional room filter
              - limit: max number of results (for specific label queries)
        """
        label_raw = (input_protocol.action or "").strip()
        room = (input_protocol.room or "").strip() or None
        limit = input_protocol.limit or self.default_limit

        if not self.sqlite_path:
            logging.error("LostAndFound connect called with no sqlite_path configured")
            return

        if not label_raw:
            logging.warning(
                "LostAndFound: empty action; expected an object label or 'all'."
            )
            return

        # Normalize action
        normalized_label = label_raw.lower()

        # If user is asking for "all objects"
        if normalized_label in {"all", "everything", "objects", "*"}:
            await self._handle_all_objects_query(room=room)
        else:
            await self._handle_label_query(
                label=normalized_label,
                room=room,
                limit=limit,
            )

    async def _handle_label_query(
        self,
        label: str,
        room: Optional[str],
        limit: int,
    ) -> None:
        """
        Fetch and log the most recent sightings for a specific label.
        """
        try:
            rows = self._query_sightings_for_label(label=label, room=room, limit=limit)
        except Exception as e:
            logging.error(f"LostAndFound: DB query failed for label '{label}': {e}")
            return

        if not rows:
            if room:
                logging.info(
                    f"LostAndFound: no sightings found for label '{label}' in room '{room}'."
                )
            else:
                logging.info(f"LostAndFound: no sightings found for label '{label}'.")
            return

        header_room = f" (room='{room}')" if room else ""
        logging.info(
            f"LostAndFound: latest {len(rows)} sighting(s) for label '{label}'{header_room}:"
        )

        for r in rows:
            ts = r["ts"]
            rm = r["room"]
            conf = r["conf"]
            crop_path = r["crop_path"]
            frame_path = r["frame_path"]
            sharp = r["sharpness"]
            logging.info(
                "  - ts=%s, room=%s, conf=%.3f, sharp=%.1f, crop=%s, frame=%s",
                ts,
                rm,
                conf,
                sharp,
                crop_path,
                frame_path,
            )

    async def _handle_all_objects_query(self, room: Optional[str]) -> None:
        """
        Fetch and log the latest sighting per label (optionally filtered by room).
        """
        try:
            rows = self._query_latest_by_label(room=room)
        except Exception as e:
            logging.error(f"LostAndFound: DB query failed for 'all' objects: {e}")
            return

        if not rows:
            if room:
                logging.info(f"LostAndFound: no objects found in room '{room}'.")
            else:
                logging.info("LostAndFound: no objects found in the database.")
            return

        header_room = f" (room='{room}')" if room else ""
        logging.info(
            f"LostAndFound: latest sighting per label{header_room} (total {len(rows)} label(s)):"
        )

        for r in rows:
            label = r["label"]
            ts = r["ts"]
            rm = r["room"]
            conf = r["conf"]
            crop_path = r["crop_path"]
            frame_path = r["frame_path"]
            logging.info(
                "  - label=%s, ts=%s, room=%s, conf=%.3f, crop=%s, frame=%s",
                label,
                ts,
                rm,
                conf,
                crop_path,
                frame_path,
            )

    def _query_sightings_for_label(
        self,
        label: str,
        room: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Query the sightings table for the most recent sightings of a given label.
        """
        with closing(
            sqlite3.connect(self.sqlite_path, check_same_thread=False)
        ) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if room:
                cur.execute(
                    """
                    SELECT
                        id, ts, room, label, conf,
                        frame_path, crop_path, sharpness, scene_path, frame_id
                    FROM sightings
                    WHERE label = ? AND room = ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (label, room, int(limit)),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id, ts, room, label, conf,
                        frame_path, crop_path, sharpness, scene_path, frame_id
                    FROM sightings
                    WHERE label = ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (label, int(limit)),
                )

            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def _query_latest_by_label(self, room: Optional[str]) -> List[Dict[str, Any]]:
        """
        Query the latest_by_label table and join with sightings to get
        one latest sighting per label.

        If `room` is provided, filter by room at the sightings level.
        """
        with closing(
            sqlite3.connect(self.sqlite_path, check_same_thread=False)
        ) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            if room:
                cur.execute(
                    """
                    SELECT
                        s.id,
                        s.ts,
                        s.room,
                        s.label,
                        s.conf,
                        s.frame_path,
                        s.crop_path,
                        s.sharpness,
                        s.scene_path,
                        s.frame_id
                    FROM latest_by_label AS l
                    JOIN sightings AS s
                      ON s.id = l.sighting_id
                    WHERE s.room = ?
                    ORDER BY s.ts DESC
                    """,
                    (room,),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        s.id,
                        s.ts,
                        s.room,
                        s.label,
                        s.conf,
                        s.frame_path,
                        s.crop_path,
                        s.sharpness,
                        s.scene_path,
                        s.frame_id
                    FROM latest_by_label AS l
                    JOIN sightings AS s
                      ON s.id = l.sighting_id
                    ORDER BY s.ts DESC
                    """
                )

            rows = cur.fetchall()
            return [dict(r) for r in rows]
