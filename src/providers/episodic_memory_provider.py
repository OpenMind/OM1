import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from providers.singleton import singleton


@singleton
class EpisodicMemoryProvider:
    """
    Singleton provider for storing and retrieving episodic memories.

    Stores robot interaction episodes persistently to disk and supports
    semantic recall via cosine similarity on OpenAI embeddings.
    """

    def __init__(self):
        self._episodes: List[Dict[str, Any]] = []
        self._pending_flush: List[Dict[str, Any]] = []
        self._openai_client: Optional[AsyncOpenAI] = None
        self._current_mode: str = "default"
        self._loaded_modes: set = set()
        self._storage_dir: str = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../config/memory")
        )
        logging.info("EpisodicMemoryProvider initialized")

    def _get_storage_path(self, mode: str) -> str:
        os.makedirs(self._storage_dir, exist_ok=True)
        return os.path.join(self._storage_dir, f".{mode}.episodes.json")

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI()
        return self._openai_client

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Embed text using OpenAI text-embedding-3-small (async)."""
        if not text or not text.strip():
            return None
        try:
            client = self._get_openai_client()
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text.strip(),
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"EpisodicMemoryProvider: embedding failed: {e}")
            return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def load_mode(self, mode: str) -> None:
        """Load episodes for a given mode from disk (public, idempotent)."""
        if mode in self._loaded_modes:
            return
        path = self._get_storage_path(mode)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    stored: List[Dict[str, Any]] = json.load(f)
                existing_ids = {ep["episode_id"] for ep in self._episodes}
                added = 0
                for ep in stored:
                    if ep.get("episode_id") not in existing_ids:
                        self._episodes.append(ep)
                        added += 1
                logging.info(
                    f"EpisodicMemoryProvider: loaded {added} episodes for mode '{mode}'"
                )
            except Exception as e:
                logging.error(
                    f"EpisodicMemoryProvider: failed to load episodes for mode '{mode}': {e}"
                )
        self._loaded_modes.add(mode)

    def save_to_disk(self) -> None:
        """Flush all pending episodes to disk (called by background writer)."""
        if not self._pending_flush:
            return

        by_mode: Dict[str, List[Dict[str, Any]]] = {}
        for ep in self._pending_flush:
            mode = ep.get("mode", "default")
            by_mode.setdefault(mode, []).append(ep)

        for mode, new_eps in by_mode.items():
            path = self._get_storage_path(mode)
            existing: List[Dict[str, Any]] = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing.extend(new_eps)
            if len(existing) > 1000:
                existing = existing[-1000:]
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
                logging.debug(
                    f"EpisodicMemoryProvider: saved {len(new_eps)} episodes for mode '{mode}'"
                )
            except Exception as e:
                logging.error(f"EpisodicMemoryProvider: failed to save: {e}")

        self._pending_flush = []

    async def write_episode(
        self,
        mode: str,
        voice_input: str,
        actions: List[Dict[str, str]],
        battery: Optional[str] = None,
    ) -> None:
        """Write a new episode to in-memory store (async)."""
        if not voice_input or not voice_input.strip():
            return
        embedding = await self._embed(voice_input)
        if embedding is None:
            return
        episode: Dict[str, Any] = {
            "episode_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "mode": mode,
            "voice_input": voice_input.strip(),
            "actions": actions,
            "battery": battery,
            "embedding": embedding,
        }
        self._episodes.append(episode)
        self._pending_flush.append(episode)
        self._current_mode = mode
        logging.debug(
            f"EpisodicMemoryProvider: wrote episode for mode '{mode}': '{voice_input[:60]}'"
        )

    async def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve the most semantically relevant past episodes for a query (async)."""
        if not query or not query.strip():
            return []

        self.load_mode(self._current_mode)

        if not self._episodes:
            return []

        query_embedding = await self._embed(query)
        if query_embedding is None:
            return []

        scored: List[tuple] = []
        for ep in self._episodes:
            ep_embedding = ep.get("embedding")
            if ep_embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, ep_embedding)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {k: v for k, v in ep.items() if k != "embedding"}
            for _, ep in scored[:top_k]
        ]
