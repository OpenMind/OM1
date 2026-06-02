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

The log may contain [User: xxx] tags indicating which user said what.
Preserve this association: if a fact comes from a tagged section,
include the user_id in your output.

For each fact, assign a category tag:
- [IDENTITY] user identity (name, age, occupation)
- [PREFERENCE] user preference (language, style, habits)
- [FACT] important facts (decisions, agreements, locations)

Output format (one per line):
- [IDENTITY] [user:alice] User's name is Alice
- [PREFERENCE] [user:bob] User prefers casual tone
- [FACT] [user:alice] User lives in Beijing
- [FACT] User asked about the weather (no user tag if unknown)

If no meaningful facts, respond with exactly "NONE"
"""

SCORE_PROMPT = """\
You are a memory manager for a robot. Evaluate each candidate fact \
against the current memory and decide what to do.

Score each candidate on three dimensions (1-5):
- durability: will this matter in future conversations?
- novelty: is this new information not already in memory?
- significance: how important is this for understanding the user?

Category must be one of: IDENTITY, PREFERENCE, FACT

Decision rules:
- Total score >= 12 AND fact is not in memory → "PROMOTE"
- Fact contradicts or updates existing memory → "UPDATE" (specify which line to replace)
- Total score < 12 or already known → "SKIP"

Current memory:
{memory}

Candidate facts:
{candidates}

Respond with a JSON array (no extra text):
[
  {{"fact": "...", "category": "IDENTITY", "user_id": "alice", "durability": 5, "novelty": 5, "significance": 4, "decision": "PROMOTE"}},
  {{"fact": "...", "category": "FACT", "user_id": null, "durability": 5, "novelty": 5, "significance": 5, "decision": "UPDATE", "replaces": "old fact text"}},
  {{"fact": "...", "category": "PREFERENCE", "user_id": "bob", "durability": 1, "novelty": 1, "significance": 1, "decision": "SKIP"}}
]

Include "user_id" if the fact is associated with a specific user, otherwise set it to null.
"""


@singleton
class MemorySummarizer:
    """Long-term memory manager.

    Stage 1: Extract candidate facts from daily logs.
    Stage 2: Score candidates against existing per-user facts.
    Stage 3: Route PROMOTE decisions to per-user facts.json.

    Parameters
    ----------
    memory_root : str or Path
        Root directory for memory storage.
    api_key : str
        API key for the LLM service.
    base_url : str
        Base URL for the LLM API endpoint.
    model : str
        LLM model to use for summarization.
    """

    SUMMARY_THRESHOLD: int = 2
    DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
    DEFAULT_BASE_URL = "https://api.openmind.com/api/core/gemini"

    def __init__(
        self,
        memory_root: str | Path,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.memory_root = Path(memory_root)
        self.daily_dir = self.memory_root / "daily"
        self.users_dir = self.memory_root / "users"
        self._marker_file = self.memory_root / ".last_summary"
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._running = False

    def check_eligibility(self) -> bool:
        """Check whether new conversation sections exceed the summarization threshold.

        Returns
        -------
        bool
            True if the number of new sections >= SUMMARY_THRESHOLD.
        """
        if self._running:
            return False
        last_summary = self._read_last_summary()
        unprocessed = self._find_unprocessed(last_summary)
        if not unprocessed:
            return False
        section_re = re.compile(r"^## \d{2}:\d{2}:\d{2}")
        count = 0
        for f in unprocessed:
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                continue
            for line in f.read_text(encoding="utf-8").split("\n"):
                match = section_re.match(line)
                if match and last_summary:
                    try:
                        t = datetime.strptime(match.group(0)[3:], "%H:%M:%S")
                        section_dt = file_date.replace(
                            hour=t.hour,
                            minute=t.minute,
                            second=t.second,
                        )
                        if section_dt > last_summary:
                            count += 1
                    except ValueError:
                        count += 1
                elif match:
                    count += 1
        return count >= self.SUMMARY_THRESHOLD

    async def run(self) -> None:
        """Execute the two-stage summarization pipeline."""
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

            log_content = self._read_files(unprocessed, last_summary)

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

    def _read_last_summary(self) -> Optional[datetime]:
        """Read the last summary timestamp from .last_summary marker file."""
        if not self._marker_file.exists():
            return None
        try:
            text = self._marker_file.read_text(encoding="utf-8").strip()
            return datetime.strptime(text, "%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            return None

    def _write_last_summary(self) -> None:
        """Write current timestamp to .last_summary marker file."""
        self._marker_file.write_text(datetime.now().strftime("%Y-%m-%d %H:%M"), encoding="utf-8")

    def _find_unprocessed(self, last_summary: Optional[datetime]) -> list[Path]:
        """Return daily log files whose date >= the last summary date.

        Parameters
        ----------
        last_summary : datetime or None
            If None, all daily files are returned.

        Returns
        -------
        list of Path
            Sorted list of daily log file paths.
        """
        if not self.daily_dir.exists():
            return []
        results: list[Path] = []
        for f in sorted(self.daily_dir.glob("*.md")):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                continue
            if last_summary is None or file_date.date() >= last_summary.date():
                results.append(f)
        return results

    @staticmethod
    def _read_files(files: list[Path], last_summary: Optional[datetime] = None) -> str:
        """Concatenate daily log files, filtering out processed sections.

        Parameters
        ----------
        files : list of Path
            Daily log files to read.
        last_summary : datetime or None
            Cutoff timestamp. If None, all sections are included.

        Returns
        -------
        str
            Combined log content for LLM processing.
        """
        parts: list[str] = []
        section_re = re.compile(r"^## (\d{2}:\d{2}:\d{2})")
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as e:
                logging.warning(f"Memory summarization: failed to read {f.name}: {e}")
                continue

            if last_summary is None:
                parts.append(content)
                continue

            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            except ValueError:
                parts.append(content)
                continue

            filtered_sections: list[str] = []
            current_section: list[str] = []
            current_keep = True
            for line in content.split("\n"):
                match = section_re.match(line)
                if match:
                    if current_keep and current_section:
                        filtered_sections.append("\n".join(current_section))
                    current_section = [line]
                    try:
                        t = datetime.strptime(match.group(1), "%H:%M:%S")
                        section_dt = file_date.replace(
                            hour=t.hour,
                            minute=t.minute,
                            second=t.second,
                        )
                        current_keep = section_dt > last_summary
                    except ValueError:
                        current_keep = True
                else:
                    current_section.append(line)

            if current_keep and current_section:
                filtered_sections.append("\n".join(current_section))

            if filtered_sections:
                parts.append("\n".join(filtered_sections))

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
            timeout=10,
        )
        result = (response.choices[0].message.content or "").strip()
        if result.upper() == "NONE":
            logging.debug("Memory summarization: no candidate facts extracted")
            return ""
        logging.debug(f"Memory summarization: extracted candidates:\n{result}")
        return result

    async def _score_candidates(self, candidates: str) -> list[dict]:
        """LLM call to score each candidate against existing per-user facts.

        Returns
        -------
        list of dict
            Each dict has keys: fact, decision, user_id, and optionally replaces.
        """
        existing = self._read_all_user_facts()

        prompt = SCORE_PROMPT.format(memory=existing, candidates=candidates)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            timeout=10,
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

    def _read_all_user_facts(self) -> str:
        """Read all per-user facts.json files and format as context for scoring."""
        if not self.users_dir.exists():
            return "(no existing facts)"

        parts = []
        for user_dir in sorted(self.users_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            facts_path = user_dir / "facts.json"
            if not facts_path.exists():
                continue
            try:
                data = json.loads(facts_path.read_text(encoding="utf-8"))
                facts_list = data.get("facts", [])
                if facts_list:
                    uid = user_dir.name
                    lines = [f"[User: {uid}]"]
                    for f in facts_list:
                        lines.append(f"- [{f.get('category', 'FACT')}] {f.get('fact', '')}")
                    parts.append("\n".join(lines))
            except Exception:
                continue

        return "\n\n".join(parts) if parts else "(no existing facts)"

    def _apply_decisions(self, decisions: list[dict]) -> None:
        """Route scored decisions to per-user facts.json.

        - **PROMOTE**: Append to user's facts.json if user_id is present.
        - **UPDATE**: Replace old fact in user's facts.json.
        - **SKIP**: No action taken.

        Parameters
        ----------
        decisions : list of dict
        """
        # Track per-user facts: user_id -> [(fact, category, replaces)]
        user_facts: dict[str, list[tuple[str, str, Optional[str]]]] = {}

        for item in decisions:
            decision = item.get("decision", "SKIP").upper()
            fact = item.get("fact", "")
            category = item.get("category", "FACT").upper()
            uid = item.get("user_id")
            if not fact or not uid:
                continue

            if decision == "PROMOTE":
                user_facts.setdefault(uid, []).append((fact, category, None))
            elif decision == "UPDATE":
                old = item.get("replaces", "")
                user_facts.setdefault(uid, []).append((fact, category, old))

        self._apply_user_facts(user_facts)

    def _apply_user_facts(self, user_facts: dict[str, list[tuple[str, str, Optional[str]]]]) -> None:
        """Append or update facts in each user's facts.json.

        Parameters
        ----------
        user_facts : dict
            Mapping of user_id -> [(fact_text, category, replaces)].
        """
        if not user_facts:
            return

        from fuser.memory_base.writer import MemoryWriter

        writer = MemoryWriter()
        now = datetime.now().isoformat(timespec="seconds")

        for uid, facts in user_facts.items():
            uid_lower = uid.strip().lower()
            if not uid_lower or uid_lower == "unknown":
                continue

            writer.ensure_user_dir(uid_lower)
            facts_path = writer.users_dir / uid_lower / "facts.json"

            try:
                data = json.loads(facts_path.read_text(encoding="utf-8"))
            except Exception:
                data = {"user_id": uid_lower, "facts": []}

            existing_texts = {f.get("fact", "") for f in data.get("facts", [])}
            added = 0
            for fact_text, category, replaces in facts:
                # Handle UPDATE: remove old fact if found
                if replaces:
                    data["facts"] = [f for f in data["facts"] if f.get("fact", "") != replaces]
                    existing_texts.discard(replaces)
                    logging.info(f"Memory UPDATE for {uid_lower}: replaced '{replaces[:40]}'")

                if fact_text not in existing_texts:
                    data["facts"].append(
                        {
                            "fact": fact_text,
                            "category": category,
                            "added_at": now,
                        }
                    )
                    existing_texts.add(fact_text)
                    added += 1

            if added or any(r for _, _, r in facts if r):
                facts_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                logging.info(f"Memory: updated facts for user {uid_lower} (+{added})")
