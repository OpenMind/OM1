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

EXPIRE_PROMPT = """\
You are a memory manager for a robot. Review the existing memory facts \
below and determine which ones should be expired.

Only review facts under ## Preferences and ## Facts sections.
Do NOT expire facts under ## Identity — those are permanent.

A fact should be marked "EXPIRED" if BOTH conditions are true:
1. EVENT-CLOSED: It describes a specific, time-bound situation
2. LOW-IDENTITY SIGNAL: It reveals nothing stable about the user — \
no recurring pattern, preference, habit, relationship, or long-term goal

Recent conversations (for context):
{recent_log}

Current memory facts to review:
{facts}

Only output facts you judge as EXPIRED.
Respond with a JSON array (no extra text):
[
  {{"fact": "...", "decision": "EXPIRED"}}
]

If no facts should be expired, respond with: []
"""


@singleton
class MemorySummarizer:
    """Long-term memory manager.

    Stage 0: Review existing facts for expiration.
    Stage 1: Extract candidate from daily logs.
    Stage 2: Score candidates against existing MEMORY.md.
    Stage 3: Execute PROMOTE / UPDATE / SKIP decisions.

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

    # Summarize when new conversations chunks is more than 2
    SUMMARY_THRESHOLD: int = 2
    EXPIRE_THRESHOLD: int = 5
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
        self.memory_file = self.memory_root / "MEMORY.md"
        self.daily_dir = self.memory_root / "daily"
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
        """Execute the three-stage summarization pipeline."""
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

            await self._review_expiration(log_content)

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
        """Parse the ``<!-- last_summary: ... -->`` marker from MEMORY.md.

        Returns
        -------
        datetime or None
            Timestamp of the last summarization run, or None if no
            marker exists or the file is missing.
        """
        if not self.memory_file.exists():
            return None
        content = self.memory_file.read_text(encoding="utf-8")
        match = self._MARKER_RE.search(content)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
        return None

    def _write_last_summary(self) -> None:
        """Insert or update the ``<!-- last_summary: ... -->`` marker."""
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

    _CATEGORY_MAP = {
        "IDENTITY": "## Identity",
        "PREFERENCE": "## Preferences",
        "FACT": "## Facts",
    }

    def _apply_decisions(self, decisions: list[dict]) -> None:
        """Execute scored decisions against MEMORY.md and per-user facts.json.

        - **PROMOTE**: Append fact under its category section in MEMORY.md,
          and also append to user's facts.json if user_id is present.
        - **UPDATE**: Replace the old fact text with the new fact.
          Falls back to PROMOTE if the old text is not found.
        - **SKIP**: No action taken.

        Parameters
        ----------
        decisions : list of dict
        """
        promoted: dict[str, list[str]] = {}  # category -> [facts]
        updated: list[tuple[str, str]] = []
        # Track per-user facts: user_id -> [(fact, category)]
        user_facts: dict[str, list[tuple[str, str]]] = {}

        for item in decisions:
            decision = item.get("decision", "SKIP").upper()
            fact = item.get("fact", "")
            category = item.get("category", "FACT").upper()
            uid = item.get("user_id")
            if not fact:
                continue

            if decision == "PROMOTE":
                promoted.setdefault(category, []).append(fact)
                if uid:
                    user_facts.setdefault(uid, []).append((fact, category))
            elif decision == "UPDATE":
                old = item.get("replaces", "")
                if old:
                    updated.append((old, fact))
                else:
                    promoted.setdefault(category, []).append(fact)
                if uid:
                    user_facts.setdefault(uid, []).append((fact, category))

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
                promoted.setdefault("FACT", []).append(new_fact)

        for category, facts in promoted.items():
            header = self._CATEGORY_MAP.get(category, "## Facts")
            header_line = header + "\n"
            new_bullets = "\n".join(f"- {f} <!-- expired: 0 -->" for f in facts) + "\n"
            if header_line in content:
                content = content.replace(header_line, header_line + new_bullets)
            else:
                content += f"\n{header}\n{new_bullets}"

        logging.info(f"Memory summarization PROMOTE: " f"{sum(len(v) for v in promoted.values())} facts")
        self._safe_write(content)

        # Route facts to per-user facts.json
        self._apply_user_facts(user_facts)

    def _safe_write(self, content: str) -> None:
        """Atomically write content to MEMORY.md via a temporary file."""
        tmp = self.memory_file.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.memory_file)

    def _apply_user_facts(self, user_facts: dict[str, list[tuple[str, str]]]) -> None:
        """Append promoted facts to each user's facts.json.

        Parameters
        ----------
        user_facts : dict
            Mapping of user_id -> [(fact_text, category)].
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
            for fact_text, category in facts:
                if fact_text not in existing_texts:
                    data["facts"].append(
                        {
                            "fact": fact_text,
                            "category": category,
                            "added_at": now,
                        }
                    )
                    added += 1

            if added:
                facts_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                logging.info(f"Memory: added {added} facts to user {uid_lower}")

    _EXPIRED_RE = re.compile(r"\s*<!-- expired: (\d+) -->$")

    @staticmethod
    def _extract_reviewable_facts(content: str) -> list[str]:
        """Extract facts from ## Preferences and ## Facts sections.

        Strips the inline ``<!-- expired: N -->`` marker before returning.
        Skips ## Identity section entirely since those are permanent.

        Returns
        -------
        list of str
            Fact texts (without leading ``- `` or expired marker).
        """
        expired_re = re.compile(r"\s*<!-- expired: \d+ -->$")
        facts: list[str] = []
        in_reviewable = False
        for line in content.split("\n"):
            if line.startswith("## "):
                section = line.strip().lower()
                in_reviewable = section in ("## preferences", "## facts")
                continue
            if in_reviewable and line.startswith("- "):
                raw = line[2:].strip()
                clean = expired_re.sub("", raw).strip()
                if clean:
                    facts.append(clean)
        return facts

    async def _review_expiration(self, recent_log: str) -> None:
        """LLM call to check existing facts for staleness."""
        if not self.memory_file.exists():
            return

        content = self.memory_file.read_text(encoding="utf-8")
        facts = self._extract_reviewable_facts(content)
        if not facts:
            return

        facts_text = "\n".join(f"- {f}" for f in facts)
        prompt = EXPIRE_PROMPT.format(
            recent_log=recent_log[:2000],
            facts=facts_text,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                timeout=10,
            )
            raw = (response.choices[0].message.content or "").strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            decisions = json.loads(cleaned)
            if not isinstance(decisions, list):
                return
        except Exception as e:
            logging.warning(f"Memory expiration review failed: {e}")
            return

        self._apply_expiration(decisions)

    def _apply_expiration(self, decisions: list[dict]) -> None:
        """Update inline expired counters and remove facts at threshold.

        EXPIRED: increment ``<!-- expired: N -->`` counter.
        If N >= EXPIRE_THRESHOLD, remove the line.
        Facts not mentioned by LLM are left unchanged.
        """
        if not self.memory_file.exists():
            return

        expired_set: set[str] = set()
        for item in decisions:
            fact = item.get("fact", "").strip()
            if fact and item.get("decision", "").upper() == "EXPIRED":
                expired_set.add(fact)

        if not expired_set:
            return

        content = self.memory_file.read_text(encoding="utf-8")
        new_lines: list[str] = []
        expired_re = self._EXPIRED_RE

        for line in content.split("\n"):
            if not line.startswith("- "):
                new_lines.append(line)
                continue

            match = expired_re.search(line)
            if match:
                count = int(match.group(1))
                fact_text = expired_re.sub("", line[2:]).strip()
            else:
                count = 0
                fact_text = line[2:].strip()

            if fact_text not in expired_set:
                new_lines.append(line)
                continue

            count += 1
            if count >= self.EXPIRE_THRESHOLD:
                logging.info(f"Memory EXPIRED (removed): '{fact_text[:60]}'")
                continue
            new_lines.append(f"- {fact_text} <!-- expired: {count} -->")

        self._safe_write("\n".join(new_lines))
