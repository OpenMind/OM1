import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import openai

from providers.singleton import singleton

EXTRACT_PROMPT = """\
Extract candidate facts from the robot-human interaction log below.

For each fact, assign a category tag:
- [IDENTITY] user identity (name, age, occupation)
- [PREFERENCE] user preference (language, style, habits)
- [FACT] important facts (decisions, agreements, locations)

Output format (one per line):
- [IDENTITY] User's name is Alice
- [PREFERENCE] User prefers casual tone
- [FACT] User lives in Beijing

If no meaningful facts, respond with exactly "NONE"
"""

SCORE_PROMPT = """\
You are a memory manager for a robot. Evaluate each candidate fact \
against the current memory and decide what to do.

Score each candidate on three dimensions (1-5):
- durability: will this matter in future conversations?
- novelty: is this new information not already in memory?
- significance: how important is this for understanding the user?

Decision rules:
- Total score >= 10 AND fact is not in memory → "PROMOTE"
- Fact contradicts or updates existing memory → "UPDATE" (specify which line to replace)
- Total score < 10 or already known → "SKIP"

Current memory:
{memory}

Candidate facts:
{candidates}

Respond with a JSON array (no extra text):
[
  {{"fact": "...", "durability": 5, "novelty": 5, "significance": 4, "decision": "PROMOTE"}},
  {{"fact": "...", "durability": 5, "novelty": 5, "significance": 5, "decision": "UPDATE", "replaces": "old fact text"}},
  {{"fact": "...", "durability": 1, "novelty": 1, "significance": 1, "decision": "SKIP"}}
]
"""


@singleton
class MemorySummarizer:
    """Long-term memory manager.

    Step 1: Extract candidate from daily logs.
    Step 2: Score candidates against existing MEMORY.md.
    Step 3: Execute PROMOTE / UPDATE / SKIP decisions.

    Parameters
    ----------
    memory_root : str or Path
        Root directory for memory storage.
    client : openai.AsyncClient
        OpenAI async client.
    model : str
        LLM model to use for summarization.
    """

    SUMMARY_INTERVAL = 50  # Summarize every 50 ticks

    def __init__(
        self,
        memory_root: str | Path,
        client: openai.AsyncClient,
        model: str,
    ):
        self.memory_root = Path(memory_root)
        self.memory_file = self.memory_root / "MEMORY.md"
        self.daily_dir = self.memory_root / "daily"
        self._client = client
        self._model = model
        self._running = False

    async def run(self) -> None:
        """Execute the pipeline.

        The main summarization thread.
        """
        if self._running:
            logging.debug("Summarizer already running, skipping")
            return

        self._running = True
        try:
            last_summary = self._read_last_summary()
            unprocessed = self._find_unprocessed(last_summary)

            if not unprocessed:
                self._write_last_summary()
                return

            log_content = self._read_files(unprocessed)

            candidates = await self._extract_candidates(log_content)
            if not candidates:
                self._write_last_summary()
                return

            decisions = await self._score_candidates(candidates)
            if not decisions:
                self._write_last_summary()
                return

            self._apply_decisions(decisions)

            self._write_last_summary()
            logging.info(
                f"Memory summarization complete: processed {len(unprocessed)} daily files, "
                f"{len(decisions)} candidates evaluated"
            )
        except asyncio.CancelledError:
            logging.info("Memory summarization cancelled")
        except Exception as e:
            logging.error(f"Memory summarization failed: {e}")
        finally:
            self._running = False

    _MARKER_RE = re.compile(r"<!-- last_summary: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) -->")

    def _read_last_summary(self) -> Optional[datetime]:
        """Parse the latest summary date from MEMORY.md."""
        if not self.memory_file.exists():
            return None
        content = self.memory_file.read_text(encoding="utf-8")
        match = self._MARKER_RE.search(content)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
        return None

    def _write_last_summary(self) -> None:
        """Update the last_summary marker in MEMORY.md."""
        if not self.memory_file.exists():
            return
        content = self.memory_file.read_text(encoding="utf-8")
        marker = f"<!-- last_summary: " f"{datetime.now().strftime('%Y-%m-%d %H:%M')} -->"
        if "<!-- last_summary:" in content:
            content = self._MARKER_RE.sub(marker, content)
        else:
            content = marker + "\n" + content
        self._safe_write(content)

    def _find_unprocessed(self, last_summary: Optional[datetime]) -> list[Path]:
        """Return daily files newer than the latest summary date."""
        if not self.daily_dir.exists():
            return []
        results: list[Path] = []
        for f in sorted(self.daily_dir.glob("*.md")):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                continue
            if last_summary is None or file_date >= last_summary:
                results.append(f)
        return results

    @staticmethod
    def _read_files(files: list[Path]) -> str:
        """Concatenate content of multiple daily files."""
        parts: list[str] = []
        for f in files:
            try:
                parts.append(f.read_text(encoding="utf-8"))
            except Exception as e:
                logging.warning(f"Dreaming: failed to read {f.name}: {e}")
        return "\n\n".join(parts)

    async def _extract_candidates(self, log: str) -> str:
        """LLM call to extract candidate facts from interaction log.

        Returns
        -------
        str
            Candidate facts as markdown bullets, or empty string.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": log},
            ],
            timeout=30,
        )
        result = (response.choices[0].message.content or "").strip()
        if result.upper() == "NONE":
            logging.debug("Memory summarization: no candidate facts extracted")
            return ""
        logging.debug(f"Memory summarization: extracted candidates:\n{result}")
        return result

    async def _score_candidates(self, candidates: str) -> list[dict]:
        """LLM call to score each candidate against existing memory.

        Returns
        -------
        list of dict
            Each dict has keys: fact, decision, and optionally replaces.
        """
        existing = ""
        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")

        prompt = SCORE_PROMPT.format(memory=existing, candidates=candidates)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()

        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            decisions = json.loads(cleaned)
            if not isinstance(decisions, list):
                logging.warning("Memory summarization: score response is not a list")
                return []
            logging.debug(f"Memory summarization: scored {len(decisions)} candidates")
            return decisions
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Memory summarization: failed to parse score response: {e}")
            return []

    def _apply_decisions(self, decisions: list[dict]) -> None:
        """Execute PROMOTE / UPDATE / SKIP on MEMORY.md."""
        promoted: list[str] = []
        updated: list[tuple[str, str]] = []

        for item in decisions:
            decision = item.get("decision", "SKIP").upper()
            fact = item.get("fact", "")
            if not fact:
                continue

            if decision == "PROMOTE":
                promoted.append(fact)
            elif decision == "UPDATE":
                old = item.get("replaces", "")
                if old:
                    updated.append((old, fact))
                else:
                    promoted.append(fact)

        if not promoted and not updated:
            return

        content = ""
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")

        for old_fact, new_fact in updated:
            if old_fact in content:
                content = content.replace(old_fact, new_fact)
                logging.info(f"Memory summarization UPDATE: '{old_fact[:40]}' → '{new_fact[:40]}'")
            else:
                promoted.append(new_fact)

        if promoted:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            section = f"\n\n## Memory summarization ({timestamp})\n"
            section += "\n".join(f"- {f}" for f in promoted) + "\n"
            content += section
            logging.info(f"Memory summarization PROMOTE: {len(promoted)} fact(s)")

        self._safe_write(content)

    def _safe_write(self, content: str) -> None:
        """Replace with tmp file."""
        tmp = self.memory_file.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.memory_file)
